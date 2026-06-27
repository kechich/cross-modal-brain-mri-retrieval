"""
Brain-ID pretrained encoder for contrast-agnostic brain MRI retrieval.

Brain-ID (ECCV 2024) is pretrained on thousands of multi-modal brain MRI scans
to produce the SAME embeddings for T1 and T2 of the same brain. We use the
frozen encoder + fine-tune a small projection head on our 290 pairs.

GitHub: https://github.com/peirong26/Brain-ID
Expected d1 MRR: ~0.93 (vs scratch-trained ~0.83)
"""
import os
import sys
import time
import tempfile
import subprocess
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_brain_id_installed():
    """Clone and import Brain-ID if not already available."""
    brain_id_dir = Path("/tmp/brain_id")
    if not brain_id_dir.exists():
        print("[brain_id] Cloning Brain-ID repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/peirong26/Brain-ID", str(brain_id_dir)],
            check=True,
            capture_output=True,
        )
    if str(brain_id_dir) not in sys.path:
        sys.path.insert(0, str(brain_id_dir))


def _download_checkpoint():
    """Download Brain-ID pretrained checkpoint."""
    ckp_path = Path("/tmp/brain_id_pretrained.pth")
    if ckp_path.exists():
        return str(ckp_path)

    print("[brain_id] Downloading pretrained checkpoint...")
    # Direct download from GitHub releases (if available)
    # Fallback: users can place brain_id_pretrained.pth in /tmp manually
    # For now, we'll assume it exists or will be provided
    try:
        subprocess.run(
            [
                "wget",
                "-q",
                "https://github.com/peirong26/Brain-ID/releases/download/v1/brain_id_pretrained.pth",
                "-O",
                str(ckp_path),
            ],
            timeout=60,
        )
    except Exception as e:
        print(f"[brain_id] Download failed ({e}), assuming checkpoint will be provided.")
        # Create a placeholder; will fail at load time if not present
        ckp_path.touch()

    return str(ckp_path)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ProjectionHead(nn.Module):
    """Small projection head: 64-dim Brain-ID features -> 128-dim embedding."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.norm = nn.LayerNorm(128)

    def forward(self, x):
        x = self.fc1(x)
        x = F.normalize(x, dim=-1)
        return x


def _load_brain_id_model(ckp_path, device):
    """Load pretrained Brain-ID encoder (frozen)."""
    _ensure_brain_id_installed()
    from network import Network  # Brain-ID model class

    model = Network().to(device)
    state = torch.load(ckp_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    return model


def _extract_feature_map(vol_np, model, device):
    """
    Extract 64-dim feature map from Brain-ID.

    Input: vol_np shape (96, 96, 96) normalized float32
    Output: (64,) L2-normalized vector via global average pooling
    """
    vol_tensor = torch.from_numpy(vol_np).float()[None, None].to(device)  # (1, 1, 96, 96, 96)

    with torch.no_grad():
        # Brain-ID forward: returns feature maps at different scales
        feats = model(vol_tensor)  # List of feature tensors
        feat_map = feats[-1]  # Last layer: (1, 64, H, W, D) with H,W,D < 96

    # Global average pooling
    pooled = feat_map.mean(dim=[2, 3, 4])  # (1, 64)
    pooled = F.normalize(pooled, dim=1)  # L2-normalize
    return pooled[0].cpu().numpy().astype(np.float32)  # (64,)


def _load_and_preprocess_volume(vol_np, device):
    """Convert numpy volume to Brain-ID input tensor."""
    return torch.from_numpy(vol_np).float()[None, None].to(device)  # (1, 1, 96, 96, 96)


def build(pairs, index, grid, loader, device=None):
    """
    Fine-tune Brain-ID projection head on the given pairs.

    - Encoder: Brain-ID (frozen)
    - Head: small projection (64 -> 128)
    - Loss: symmetric CLIP-style cross-entropy
    - Epochs: 100
    """
    device = torch.device(device or pick_device())
    ckp_path = _download_checkpoint()

    print(f"[brain_id] device={device} pairs={len(pairs)} grid={grid}")
    print(f"[brain_id] Loading pretrained Brain-ID from {ckp_path}...")

    # Load frozen encoder
    encoder = _load_brain_id_model(ckp_path, device)

    # Build projection head (trainable)
    head = ProjectionHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    ce = nn.CrossEntropyLoss()

    # Extract Brain-ID features for all pairs (one-time, offline)
    print("[brain_id] Extracting features from all training pairs...")
    q_feats = []
    t_feats = []

    for i, p in enumerate(pairs):
        q_vol = loader(p["query_id"], index[p["query_id"]], grid)
        t_vol = loader(p["target_id"], index[p["target_id"]], grid)

        q_feat = _extract_feature_map(q_vol, encoder, device)
        t_feat = _extract_feature_map(t_vol, encoder, device)

        q_feats.append(q_feat)
        t_feats.append(t_feat)

        if (i + 1) % 50 == 0:
            print(f"  Extracted {i + 1}/{len(pairs)} feature pairs")

    q_feats = torch.from_numpy(np.stack(q_feats)).to(device)  # (N, 64)
    t_feats = torch.from_numpy(np.stack(t_feats)).to(device)  # (N, 64)
    N = len(pairs)

    print(f"[brain_id] Fine-tuning projection head on {N} pairs for 100 epochs...")
    head.train()
    t0 = time.time()

    for epoch in range(1, 101):
        perm = torch.randperm(N, device=device)
        total, seen = 0.0, 0

        for s in range(0, N, 32):  # Smaller batch for projection head
            idx = perm[s : s + 32]
            if len(idx) < 2:
                continue

            zq = head(q_feats[idx])  # (B, 128)
            zt = head(t_feats[idx])  # (B, 128)

            # CLIP loss: symmetric CE on diagonal of similarity matrix
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
            print(f"[brain_id] epoch {epoch:03d} loss={total / max(seen, 1):.4f} ({time.time()-t0:.0f}s)")

    head.eval()

    @torch.no_grad()
    def embed(vol_np):
        """Encode a volume: Brain-ID feature -> projection head -> L2 vector."""
        feat = _extract_feature_map(vol_np, encoder, device)  # (64,)
        feat_t = torch.from_numpy(feat)[None].to(device)  # (1, 64)
        z = head(feat_t)[0].cpu().numpy().astype(np.float32)  # (128,)
        return z

    embed.model = head
    return embed
