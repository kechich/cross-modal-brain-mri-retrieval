"""
Deformation-augmented learned embedder for the cross-modal retrieval harness.

A small shared 3-D CNN encoder trained CLIP-style on dataset1's labelled pairs, with
heavy GPU-side augmentation applied INDEPENDENTLY to the query and target each step:

  * geometric : random rigid (rotation + translation + scale) + elastic warp
                -> teaches deformation invariance directly (mimics dataset2).
  * intensity : random gamma, contrast inversion, bias field, noise
                -> bridges the ceT1<->T2 modality gap and scanner/domain shift (helps d3).

One SHARED encoder maps both modalities into a common space (better than two towers for
~290 pairs). It exposes the harness embedder contract: volume(np.float32 HxWxD) -> L2 vector.

The harness calls `build(...)` once to train, then scores the returned `embed` fn on all
three proxy levels. Trains on the pairs NOT held out for validation, so no leakage.

Tunables via env: EMB_EPOCHS, EMB_BATCH, EMB_DIM, EMB_WIDTH, EMB_LR, EMB_ROT, EMB_ELASTIC.
"""
import os
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- config
def _cfg():
    # backbone: "cnn" (from-scratch, fast) or "swin" (MONAI SwinUNETR, SSL-pretrained)
    swin = os.environ.get("EMB_BACKBONE", "cnn") == "swin"
    return dict(
        backbone="swin" if swin else "cnn",
        epochs=int(os.environ.get("EMB_EPOCHS", "80" if swin else "300")),
        batch=int(os.environ.get("EMB_BATCH", "48" if swin else "96")),
        dim=int(os.environ.get("EMB_DIM", "128")),
        width=int(os.environ.get("EMB_WIDTH", "16")),
        lr=float(os.environ.get("EMB_LR", "2e-4" if swin else "3e-4")),       # backbone lr
        lr_head=float(os.environ.get("EMB_LR_HEAD", "1e-3")),                  # new-head lr
        swin_checkpoint=os.environ.get("EMB_SWIN_CHECKPOINT", "0") == "1",     # 192GB -> off
        rot_deg=float(os.environ.get("EMB_ROT", "15")),
        elastic=float(os.environ.get("EMB_ELASTIC", "0.06")),
        weight_decay=float(os.environ.get("EMB_WD", "1e-2")),
    )


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------- model
class Encoder3D(nn.Module):
    """Compact 3-D CNN: 96 -> 48 -> 24 -> 12 -> 6, global-pool -> embedding."""

    def __init__(self, dim=128, width=16):
        super().__init__()

        def block(ci, co, stride):
            g = max(1, min(8, co))
            return nn.Sequential(
                nn.Conv3d(ci, co, 3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(g, co), nn.ReLU(inplace=True),
                nn.Conv3d(co, co, 3, padding=1, bias=False),
                nn.GroupNorm(g, co), nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(1, width, 2),
            block(width, width * 2, 2),
            block(width * 2, width * 4, 2),
            block(width * 4, width * 8, 2),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(width * 8, dim),
        )

    def forward(self, x):
        return self.net(x)


# SSL-pretrained Swin UNETR encoder weights (self-supervised on ~5k 3D CT/MRI volumes).
SSL_URL = ("https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/"
           "0.8.1/model_swinvit.pt")


def _ensure_swin_weights():
    path = Path(os.environ.get("SWIN_WEIGHTS", "model_swinvit.pt"))
    if not path.exists():
        import urllib.request
        print(f"[swin] downloading SSL weights -> {path}")
        urllib.request.urlretrieve(SSL_URL, path)
    return path


class SwinEmbed(nn.Module):
    """MONAI SwinUNETR swinViT backbone (SSL-pretrained) -> global-pool -> projection."""

    def __init__(self, dim, grid, feature_size=48):
        super().__init__()
        from monai.networks.nets import SwinUNETR
        kw = dict(in_channels=1, out_channels=1, feature_size=feature_size, use_checkpoint=True)
        try:
            net = SwinUNETR(img_size=(grid,) * 3, spatial_dims=3, **kw)
        except TypeError:                       # newer MONAI dropped img_size
            net = SwinUNETR(spatial_dims=3, **kw)
        ckpt = torch.load(_ensure_swin_weights(), map_location="cpu")
        try:
            net.load_from(weights=ckpt)
            print("[swin] SSL weights loaded via load_from")
        except Exception as e:                  # fallback: load straight into swinViT
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            sd = {k.split("swinViT.")[-1].replace("module.", ""): v for k, v in sd.items()}
            m, u = net.swinViT.load_state_dict(sd, strict=False)
            print(f"[swin] load_from failed ({type(e).__name__}); direct load "
                  f"missing={len(m)} unexpected={len(u)}")
        self.swin = net.swinViT
        with torch.no_grad():                   # infer deepest-feature channel count
            c = self._feat(torch.zeros(1, 1, grid, grid, grid)).shape[1]
        self.head = nn.Sequential(nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(c, dim))

    def _feat(self, x):
        out = self.swin(x)
        return out[-1] if isinstance(out, (list, tuple)) else out

    def forward(self, x):
        return self.head(self._feat(x))


def make_encoder(cfg, grid):
    if cfg["backbone"] == "swin":
        if grid % 32 != 0:
            raise ValueError(f"Swin needs GRID divisible by 32 (got {grid}; use 96)")
        return SwinEmbed(cfg["dim"], grid)
    return Encoder3D(cfg["dim"], cfg["width"])


class CLIPModel(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.enc = encoder
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode(self, x):
        return F.normalize(self.enc(x), dim=1)


# --------------------------------------------------------------------------- augmentation (GPU)
def _rotation_matrices(B, max_deg, device):
    a = (torch.rand(B, 3, device=device) * 2 - 1) * math.radians(max_deg)
    cz, sz = torch.cos(a[:, 0]), torch.sin(a[:, 0])
    cy, sy = torch.cos(a[:, 1]), torch.sin(a[:, 1])
    cx, sx = torch.cos(a[:, 2]), torch.sin(a[:, 2])
    z = torch.zeros(B, device=device)
    o = torch.ones(B, device=device)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(B, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(B, 3, 3)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(B, 3, 3)
    return Rz @ Ry @ Rx


def _geom_augment(x, cfg):
    """Independent rigid + elastic warp on a batch (B,1,D,H,W) via grid_sample."""
    B = x.shape[0]
    dev = x.device
    size = x.shape[2:]
    R = _rotation_matrices(B, cfg["rot_deg"], dev)
    scale = 1 + (torch.rand(B, 3, device=dev) * 2 - 1) * 0.10
    R = R * scale[:, None, :]
    trans = (torch.rand(B, 3, device=dev) * 2 - 1) * 0.12          # normalised units
    theta = torch.cat([R, trans[:, :, None]], dim=2)               # (B,3,4)
    grid = F.affine_grid(theta, (B, 1, *size), align_corners=False)
    # elastic: low-res random field upsampled to full grid
    ctrl = torch.randn(B, 3, 5, 5, 5, device=dev)
    disp = F.interpolate(ctrl, size=size, mode="trilinear", align_corners=False)
    disp = disp.permute(0, 2, 3, 4, 1) * cfg["elastic"]
    grid = grid + disp
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def _intensity_augment(x):
    """Random gamma, contrast inversion, bias field, noise — keeps background at 0."""
    B = x.shape[0]
    dev = x.device
    mask = (x > 0.02).float()
    # gamma
    gamma = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.6, 1.6)
    x = x.clamp(0, 1) ** gamma
    # contrast inversion on the foreground (the big ceT1<->T2 cue)
    inv = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.5).float()
    x = (1 - inv) * x + inv * ((1.0 - x) * mask)
    # smooth low-frequency multiplicative bias field
    bias = F.interpolate(torch.randn(B, 1, 4, 4, 4, device=dev), size=x.shape[2:],
                         mode="trilinear", align_corners=False)
    bias = 0.7 + 0.6 * torch.sigmoid(bias)
    x = x * bias
    # noise
    x = x + torch.randn_like(x) * 0.03
    x = (x * mask).clamp(0, None)
    m = x.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    return x / m


def augment(x, cfg):
    return _intensity_augment(_geom_augment(x, cfg))


# --------------------------------------------------------------------------- train / build
def _load_stack(pairs, index, grid, loader, key, device):
    vols = [loader(p[key], index[p[key]], grid) for p in pairs]
    arr = np.stack(vols).astype(np.float32)[:, None]   # (N,1,D,H,W)
    return torch.from_numpy(arr).to(device)


def build(pairs, index, grid, loader, device=None):
    """Train the embedder on `pairs` (the non-validation split) and return an embed fn."""
    cfg = _cfg()
    device = torch.device(device or pick_device())
    print(f"[learned] device={device} pairs={len(pairs)} cfg={cfg}")

    q = _load_stack(pairs, index, grid, loader, "query_id", device)
    t = _load_stack(pairs, index, grid, loader, "target_id", device)
    N = len(pairs)

    model = CLIPModel(make_encoder(cfg, grid)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    ce = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        perm = torch.randperm(N, device=device)
        total, seen = 0.0, 0
        for s in range(0, N, cfg["batch"]):
            idx = perm[s:s + cfg["batch"]]
            if len(idx) < 2:
                continue
            qa = augment(q[idx], cfg)        # independent draws -> q and t differ
            ta = augment(t[idx], cfg)
            zq, zt = model.encode(qa), model.encode(ta)
            scale = model.logit_scale.exp().clamp(max=100)
            logits = scale * zq @ zt.t()
            labels = torch.arange(len(idx), device=device)
            loss = (ce(logits, labels) + ce(logits.t(), labels)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
            seen += len(idx)
        if epoch % 25 == 0 or epoch == 1:
            print(f"[learned] epoch {epoch:03d} loss={total / max(seen,1):.4f} "
                  f"({time.time()-t0:.0f}s)")
    model.eval()

    @torch.no_grad()
    def embed(vol_np):
        x = torch.from_numpy(np.ascontiguousarray(vol_np))[None, None].float().to(device)
        z = model.encode(x)[0].cpu().numpy().astype(np.float32)
        return z

    embed.model = model        # expose for reuse / TTA later
    return embed
