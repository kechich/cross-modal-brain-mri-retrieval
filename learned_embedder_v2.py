"""
Improved embedder: v1 architecture + v2 improvements.

Keep the proven CLIP loss from v1, but add:
- Stronger geometric augmentation: rot 15°→25°, elastic 0.06→0.10, flips
- Larger network: width 16→24
- Learned temperature scale (like v1)
"""
import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg():
    return dict(
        epochs=int(os.environ.get("EMB_EPOCHS", "300")),
        batch=int(os.environ.get("EMB_BATCH", "96")),
        dim=int(os.environ.get("EMB_DIM", "128")),
        width=int(os.environ.get("EMB_WIDTH", "24")),  # ← 16→24
        lr=float(os.environ.get("EMB_LR", "3e-4")),
        rot_deg=float(os.environ.get("EMB_ROT", "25")),  # ← 15→25
        elastic=float(os.environ.get("EMB_ELASTIC", "0.10")),  # ← 0.06→0.10
        weight_decay=float(os.environ.get("EMB_WD", "1e-2")),
    )


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Encoder3D(nn.Module):
    """3-D CNN: 96 → 48 → 24 → 12 → 6, global-pool → embedding."""

    def __init__(self, dim=128, width=24):
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


class CLIPModel(nn.Module):
    def __init__(self, dim=128, width=24):
        super().__init__()
        self.enc = Encoder3D(dim, width)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode(self, x):
        return F.normalize(self.enc(x), dim=1)


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
    """Independent rigid + elastic warp + random flip (stronger)."""
    B = x.shape[0]
    dev = x.device
    size = x.shape[2:]

    # Random flip (50% per axis)
    for axis in [2, 3, 4]:
        mask = torch.rand(B, device=dev) < 0.5
        if mask.any():
            x[mask] = torch.flip(x[mask], dims=[axis])

    # Rotation + scale
    R = _rotation_matrices(B, cfg["rot_deg"], dev)
    scale = 1 + (torch.rand(B, 3, device=dev) * 2 - 1) * 0.15
    R = R * scale[:, None, :]
    trans = (torch.rand(B, 3, device=dev) * 2 - 1) * 0.15
    theta = torch.cat([R, trans[:, :, None]], dim=2)
    grid = F.affine_grid(theta, (B, 1, *size), align_corners=False)

    # Elastic warping (stronger)
    ctrl = torch.randn(B, 3, 5, 5, 5, device=dev)
    disp = F.interpolate(ctrl, size=size, mode="trilinear", align_corners=False)
    disp = disp.permute(0, 2, 3, 4, 1) * cfg["elastic"]
    grid = grid + disp
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def _intensity_augment(x):
    """Gamma, contrast inversion, bias field, noise."""
    B = x.shape[0]
    dev = x.device
    mask = (x > 0.02).float()

    # Gamma (wider)
    gamma = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.5, 1.7)
    x = x.clamp(0, 1) ** gamma

    # Contrast inversion
    inv = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.5).float()
    x = (1 - inv) * x + inv * ((1.0 - x) * mask)

    # Bias field
    bias = F.interpolate(torch.randn(B, 1, 4, 4, 4, device=dev), size=x.shape[2:],
                         mode="trilinear", align_corners=False)
    bias = 0.6 + 0.8 * torch.sigmoid(bias)
    x = x * bias

    # Noise
    x = x + torch.randn_like(x) * 0.04
    x = (x * mask).clamp(0, None)
    m = x.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    return x / m


def augment(x, cfg):
    return _intensity_augment(_geom_augment(x, cfg))


def _load_stack(pairs, index, grid, loader, key, device):
    vols = [loader(p[key], index[p[key]], grid) for p in pairs]
    arr = np.stack(vols).astype(np.float32)[:, None]
    return torch.from_numpy(arr).to(device)


def build(pairs, index, grid, loader, device=None):
    """Train on `pairs` with stronger augmentation + larger network."""
    cfg = _cfg()
    device = torch.device(device or pick_device())
    print(f"[learned_improved] device={device} pairs={len(pairs)} cfg={cfg}")

    q = _load_stack(pairs, index, grid, loader, "query_id", device)
    t = _load_stack(pairs, index, grid, loader, "target_id", device)
    N = len(pairs)

    model = CLIPModel(cfg["dim"], cfg["width"]).to(device)
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

            qa = augment(q[idx], cfg)
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
            print(f"[learned_improved] epoch {epoch:03d} loss={total / max(seen,1):.4f} "
                  f"({time.time()-t0:.0f}s)")

    model.eval()

    @torch.no_grad()
    def embed(vol_np):
        x = torch.from_numpy(np.ascontiguousarray(vol_np))[None, None].float().to(device)
        z = model.encode(x)[0].cpu().numpy().astype(np.float32)
        return z

    embed.model = model
    return embed
