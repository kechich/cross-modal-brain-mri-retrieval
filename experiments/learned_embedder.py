"""
Raw-intensity CLIP embedder — improved over v1.

Same proven approach (raw voxels → 3D CNN → CLIP loss) with targeted improvements:
  * width 32 (vs 16): 2x capacity for better feature extraction
  * rot ±25°, elastic 0.10, random flips: stronger geometric aug for d2 robustness
  * resection augmentation (p=0.4): zero a random ellipsoid to simulate post-surgery d3
  * TTA=8 at inference: average 8 augmented-view embeddings, reduces d2/d3 variance

Intensity augmentation (gamma, inversion, bias, noise) is kept — the 50% contrast
inversion is the main bridge across the ceT1↔T2 modality gap.

Harness contract: build(pairs, index, grid, loader) -> embed fn.
Tunables: EMB_EPOCHS(300), EMB_BATCH(64), EMB_DIM(128), EMB_WIDTH(32),
          EMB_LR(3e-4), EMB_ROT(25), EMB_ELASTIC(0.10), EMB_TTA(8).
"""
import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ config
def _cfg():
    return dict(
        epochs=int(os.environ.get("EMB_EPOCHS", "300")),
        batch=int(os.environ.get("EMB_BATCH", "96")),
        dim=int(os.environ.get("EMB_DIM", "128")),
        width=int(os.environ.get("EMB_WIDTH", "32")),
        lr=float(os.environ.get("EMB_LR", "3e-4")),
        rot_deg=float(os.environ.get("EMB_ROT", "25")),
        elastic=float(os.environ.get("EMB_ELASTIC", "0.10")),
        weight_decay=float(os.environ.get("EMB_WD", "1e-2")),
        resect_p=float(os.environ.get("EMB_RESECT", "0.4")),
        tta=int(os.environ.get("EMB_TTA", "8")),
    )


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------ model
class Encoder3D(nn.Module):
    """3-D CNN: 96→48→24→12→6, global-pool → embedding."""

    def __init__(self, dim=128, width=32):
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
    def __init__(self, dim=128, width=32):
        super().__init__()
        self.enc = Encoder3D(dim, width)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode(self, x):
        return F.normalize(self.enc(x), dim=1)


# ------------------------------------------------------------------ augmentation
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
    """Rigid (rot + scale + trans) + elastic warp. x: (B,1,D,H,W).

    No random flips: query and target are augmented independently, so a 50% flip
    on each axis means 87.5% of positive pairs would have mismatched orientations
    and look like negatives → CLIP loss gets pinned at log(batch_size) forever.
    """
    B, _, *size = x.shape
    dev = x.device

    R = _rotation_matrices(B, cfg["rot_deg"], dev)
    scale = 1 + (torch.rand(B, 3, device=dev) * 2 - 1) * 0.15
    R = R * scale[:, None, :]
    trans = (torch.rand(B, 3, device=dev) * 2 - 1) * 0.15
    theta = torch.cat([R, trans[:, :, None]], dim=2)
    grid = F.affine_grid(theta, (B, 1, *size), align_corners=False)
    ctrl = torch.randn(B, 3, 5, 5, 5, device=dev)
    disp = F.interpolate(ctrl, size=size, mode="trilinear", align_corners=False)
    disp = disp.permute(0, 2, 3, 4, 1) * cfg["elastic"]
    grid = grid + disp
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")


def _intensity_augment(x):
    """Gamma, contrast inversion (ceT1↔T2 bridge), bias field, noise."""
    B, _, *size = x.shape
    dev = x.device
    mask = (x > 0.02).float()

    gamma = torch.empty(B, 1, 1, 1, 1, device=dev).uniform_(0.5, 1.7)
    x = x.clamp(0, 1) ** gamma

    inv = (torch.rand(B, 1, 1, 1, 1, device=dev) < 0.5).float()
    x = (1 - inv) * x + inv * ((1.0 - x) * mask)

    bias = F.interpolate(torch.randn(B, 1, 4, 4, 4, device=dev),
                         size=size, mode="trilinear", align_corners=False)
    bias = 0.6 + 0.8 * torch.sigmoid(bias)
    x = x * bias

    x = x + torch.randn_like(x) * 0.04
    x = (x * mask).clamp(0, None)
    m = x.amax(dim=(2, 3, 4), keepdim=True).clamp_min(1e-6)
    return x / m


def _resection_augment(x, p):
    """Zero a random ellipsoid with probability p — simulates d3 post-surgery cavity."""
    B, _, D, H, W = x.shape
    dev = x.device
    do = (torch.rand(B, device=dev) < p).float().view(B, 1, 1, 1)
    dims = torch.tensor([D, H, W], device=dev, dtype=torch.float32)
    c = (torch.rand(B, 3, device=dev) * 0.5 + 0.25) * dims
    rad = (torch.rand(B, 3, device=dev) * 0.10 + 0.06) * D
    zz = torch.arange(D, device=dev).view(1, D, 1, 1)
    yy = torch.arange(H, device=dev).view(1, 1, H, 1)
    xx = torch.arange(W, device=dev).view(1, 1, 1, W)
    ell = (((zz - c[:, 0].view(B, 1, 1, 1)) / rad[:, 0].view(B, 1, 1, 1)) ** 2
         + ((yy - c[:, 1].view(B, 1, 1, 1)) / rad[:, 1].view(B, 1, 1, 1)) ** 2
         + ((xx - c[:, 2].view(B, 1, 1, 1)) / rad[:, 2].view(B, 1, 1, 1)) ** 2) <= 1.0
    keep = 1.0 - (ell.float() * do)
    return x * keep.unsqueeze(1)


def augment(x, cfg):
    x = _geom_augment(x, cfg)
    x = _intensity_augment(x)
    x = _resection_augment(x, cfg["resect_p"])
    return x


def augment_tta(x, cfg):
    """Lighter augmentation for TTA inference — no resection."""
    x = _geom_augment(x, cfg)
    x = _intensity_augment(x)
    return x


# ------------------------------------------------------------------ data loading
def _load_stack(pairs, index, grid, loader, key, device):
    vols = [loader(p[key], index[p[key]], grid) for p in pairs]
    arr = np.stack(vols).astype(np.float32)[:, None]
    return torch.from_numpy(arr).to(device)


# ------------------------------------------------------------------ train / build
def build(pairs, index, grid, loader, device=None):
    """Train and return embed fn with TTA."""
    cfg = _cfg()
    device = torch.device(device or pick_device())
    print(f"[clip] device={device}  pairs={len(pairs)}  cfg={cfg}")

    q = _load_stack(pairs, index, grid, loader, "query_id", device)
    t = _load_stack(pairs, index, grid, loader, "target_id", device)
    N = len(pairs)

    model = CLIPModel(cfg["dim"], cfg["width"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    ce = nn.CrossEntropyLoss()
    model.train()
    t0 = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        perm = torch.randperm(N, device=device)
        total, seen = 0.0, 0

        for s in range(0, N, cfg["batch"]):
            idx = perm[s : s + cfg["batch"]]
            if len(idx) < 2:
                continue

            zq = model.encode(augment(q[idx], cfg))
            zt = model.encode(augment(t[idx], cfg))

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
            print(f"[clip] epoch {epoch:03d}  loss={total / max(seen, 1):.4f}"
                  f"  ({time.time() - t0:.0f}s)")

    model.eval()
    n_tta = cfg["tta"]

    @torch.no_grad()
    def embed(vol_np):
        x = torch.from_numpy(np.ascontiguousarray(vol_np))[None, None].float().to(device)
        views = [model.encode(x)]
        for _ in range(n_tta - 1):
            views.append(model.encode(augment_tta(x, cfg)))
        z = F.normalize(torch.stack(views).mean(0), dim=1)
        return z[0].cpu().numpy().astype(np.float32)

    embed.model = model
    return embed
