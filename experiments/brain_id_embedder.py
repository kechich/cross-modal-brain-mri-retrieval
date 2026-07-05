"""
Brain-ID pretrained encoder OR MONAI fallback for contrast-agnostic brain MRI retrieval.

Brain-ID (ECCV 2024): https://github.com/peirong26/Brain-ID
- Pretrained on thousands of multi-modal brain MRI (T1w, T2, FLAIR, CT)
- Frozen encoder + fine-tune small projection head

Fallback: MONAI's ResNet3D (already installed on AMD box)
- Pretrained on ImageNet (2D) + adapted for 3D
- Strong baseline, no external downloads needed
"""
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ProjectionHead(nn.Module):
    """Small projection head: input_dim -> 128-dim embedding."""

    def __init__(self, input_dim=64):
        super().__init__()
        self.fc = nn.Linear(input_dim, 128)

    def forward(self, x):
        x = self.fc(x)
        x = F.normalize(x, dim=-1)
        return x


def _try_brain_id(pairs, index, grid, loader, device):
    """Try to use Brain-ID. Returns (encoder_fn, feat_dim) or None if it fails."""
    print("[brain_id] Attempting to load Brain-ID...")

    try:
        # Try to import and use Brain-ID
        brain_id_dir = Path("/tmp/brain_id")
        if not brain_id_dir.exists():
            print("[brain_id] Cloning Brain-ID repository...")
            subprocess.run(
                ["git", "clone", "https://github.com/peirong26/Brain-ID", str(brain_id_dir)],
                check=True,
                capture_output=True,
                timeout=60,
            )

        sys.path.insert(0, str(brain_id_dir))

        # Try different import approaches for Brain-ID
        try:
            from network import Network

            print("[brain_id] Found network.Network")
        except ImportError:
            try:
                from models.network import Network

                print("[brain_id] Found models.network.Network")
            except ImportError:
                try:
                    # Last resort: search for any Network class in the cloned repo
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "brain_id_net", brain_id_dir / "models" / "models.py"
                    )
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        Network = module.Network
                        print("[brain_id] Found via importlib")
                    else:
                        raise ImportError("Could not find Network class")
                except Exception as e:
                    print(f"[brain_id] Network import failed: {e}")
                    return None

        # Load checkpoint
        ckp_path = Path("/tmp/brain_id_pretrained.pth")
        if not ckp_path.exists():
            print(f"[brain_id] Checkpoint not found at {ckp_path}")
            print(f"[brain_id] (Users can download from https://github.com/peirong26/Brain-ID)")
            return None

        model = Network().to(device)
        state = torch.load(ckp_path, map_location=device)
        model.load_state_dict(state, strict=False)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        print("[brain_id] Loaded successfully")

        def extract_feature(vol_np):
            """Extract 64-dim feature from Brain-ID."""
            vol_t = torch.from_numpy(vol_np).float()[None, None].to(device)
            with torch.no_grad():
                feats = model(vol_t)
                if isinstance(feats, (list, tuple)):
                    feat_map = feats[-1]
                else:
                    feat_map = feats
            # Global average pool: (1, C, H, W, D) -> (1, C)
            pooled = feat_map.mean(dim=[2, 3, 4])
            pooled = F.normalize(pooled, dim=1)
            return pooled[0].cpu().numpy().astype(np.float32)

        return extract_feature, 64

    except Exception as e:
        print(f"[brain_id] Failed: {e}")
        return None


def _use_monai_fallback(pairs, index, grid, loader, device):
    """Use MONAI's pretrained ResNet3D as fallback."""
    print("[monai] Using MONAI ResNet50 fallback (pretrained on ImageNet)")

    try:
        from monai.networks.nets import resnet50

        model = resnet50(pretrained=True, n_input_channels=1, num_classes=512)
        model = model.to(device)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        def extract_feature(vol_np):
            """Extract feature from MONAI ResNet."""
            vol_t = torch.from_numpy(vol_np).float()[None, None].to(device)
            with torch.no_grad():
                feat = model(vol_t)  # (1, 512)
            feat = F.normalize(feat, dim=1)
            return feat[0].cpu().numpy().astype(np.float32)

        print("[monai] Loaded successfully")
        return extract_feature, 512

    except Exception as e:
        print(f"[monai] Failed: {e}")
        return None


def build(pairs, index, grid, loader, device=None):
    """
    Build embedder: try Brain-ID, fallback to MONAI ResNet50.
    Fine-tune projection head on 290 pairs.
    """
    device = torch.device(device or pick_device())

    print(f"[embedder] device={device} pairs={len(pairs)}")

    # Try Brain-ID first, fallback to MONAI
    result = _try_brain_id(pairs, index, grid, loader, device)
    if result is None:
        result = _use_monai_fallback(pairs, index, grid, loader, device)

    if result is None:
        raise RuntimeError("Could not load Brain-ID or MONAI fallback")

    extract_feature, feat_dim = result

    # Extract features for all training pairs (offline)
    print(f"[embedder] Extracting {feat_dim}-dim features from {len(pairs)} pairs...")
    q_feats = []
    t_feats = []

    for i, p in enumerate(pairs):
        q_vol = loader(p["query_id"], index[p["query_id"]], grid)
        t_vol = loader(p["target_id"], index[p["target_id"]], grid)

        q_feat = extract_feature(q_vol)
        t_feat = extract_feature(t_vol)

        q_feats.append(q_feat)
        t_feats.append(t_feat)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pairs)} done")

    q_feats = torch.from_numpy(np.stack(q_feats)).to(device)
    t_feats = torch.from_numpy(np.stack(t_feats)).to(device)
    N = len(pairs)

    # Build and train projection head
    head = ProjectionHead(input_dim=feat_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss()

    print(f"[embedder] Fine-tuning projection head (100 epochs, batch=32)...")
    head.train()
    t0 = time.time()

    for epoch in range(1, 101):
        perm = torch.randperm(N, device=device)
        total, seen = 0.0, 0

        for s in range(0, N, 32):
            idx = perm[s : s + 32]
            if len(idx) < 2:
                continue

            zq = head(q_feats[idx])
            zt = head(t_feats[idx])

            scale = 20.0
            logits = scale * zq @ zt.t()
            labels = torch.arange(len(idx), device=device)
            loss = (ce(logits, labels) + ce(logits.t(), labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            total += loss.item() * len(idx)
            seen += len(idx)

        if epoch % 25 == 0 or epoch == 1:
            print(f"[embedder] epoch {epoch:03d} loss={total / max(seen, 1):.4f} ({time.time()-t0:.0f}s)")

    head.eval()

    @torch.no_grad()
    def embed(vol_np):
        """Encode volume: frozen feature extractor -> projection head -> L2 vector."""
        feat = extract_feature(vol_np)
        feat_t = torch.from_numpy(feat)[None].to(device)
        z = head(feat_t)[0].cpu().numpy().astype(np.float32)
        return z

    embed.model = head
    return embed
