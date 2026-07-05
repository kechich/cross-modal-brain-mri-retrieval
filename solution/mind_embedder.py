"""MIND-input contrastive embedder — the learned model for dataset2/3.

Why this design (the two hard parts of the challenge, attacked at the right layer):

  1. MODALITY GAP (ceT1 vs T2): we do NOT feed raw intensities. We feed the **MIND descriptor
     field** (Heinrich) — local self-similarity structure that is modality-INVARIANT by
     construction. So MIND(ceT1) ≈ MIND(T2) for the same anatomy, and the encoder never has to
     learn the intensity relationship. (This is also why the training-free dense_mind already
     hits 0.71 on d1.)

  2. DEFORMATION / STRUCTURAL CHANGE (d2/d3): a 3-D CNN is trained CLIP-style so that the SAME
     patient's query and target embed together while different patients repel — with heavy,
     INDEPENDENT augmentation per side that simulates exactly what d2/d3 do: rigid+elastic warp
     (d2) and resection cavity + bias/intensity shift (d3). The network learns a *global*
     embedding that is robust to those, so it works when query and gallery no longer share a grid
     (where the dense alignment used on d1 collapses).

Plugs into the eval harness as the `mind_learned` embedder: volume -> L2 vector. Trains on the
non-validation pairs only (no leakage). Tunables via env: MIND_EPOCHS, MIND_BATCH, MIND_DIM,
MIND_WIDTH, MIND_LR, MIND_ROT, MIND_ELASTIC, MIND_RESECT, MIND_DIL, MIND_TTA.
"""
import os, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import learned_embedder as le          # reuse geometric + intensity augmentation, _load_stack
import rankers as rk                   # reuse the torch MIND field (_mind)


def _cfg():
    return dict(
        epochs=int(os.environ.get("MIND_EPOCHS", "200")),
        batch=int(os.environ.get("MIND_BATCH", "64")),
        dim=int(os.environ.get("MIND_DIM", "128")),
        width=int(os.environ.get("MIND_WIDTH", "24")),
        lr=float(os.environ.get("MIND_LR", "3e-4")),
        rot=float(os.environ.get("MIND_ROT", "20")),
        elastic=float(os.environ.get("MIND_ELASTIC", "0.08")),
        resect=float(os.environ.get("MIND_RESECT", "0.5")),     # P(resection cavity) per sample
        wd=float(os.environ.get("MIND_WD", "1e-2")),
        dil=int(os.environ.get("MIND_DIL", "2")),               # MIND dilation
        tta=int(os.environ.get("MIND_TTA", "8")),               # inference augmented views
    )


class Encoder(nn.Module):
    """Compact 3-D CNN over a 6-channel MIND field: 96->48->24->12->6, global-pool -> embedding."""

    def __init__(self, cin=6, dim=128, width=24):
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
            block(cin, width, 2),
            block(width, width * 2, 2),
            block(width * 2, width * 4, 2),
            block(width * 4, width * 8, 2),
            nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            nn.Linear(width * 8, dim),
        )

    def forward(self, x):
        return self.net(x)


def _resection(x, p):
    """Vectorised: with prob p per sample, zero a random ellipsoid in the brain (mimics d3)."""
    B, _, D, H, W = x.shape
    dev = x.device
    do = (torch.rand(B, device=dev) < p).float().view(B, 1, 1, 1)
    dims = torch.tensor([D, H, W], device=dev, dtype=torch.float32)
    c = (torch.rand(B, 3, device=dev) * 0.5 + 0.25) * dims          # central-ish centre
    rad = (torch.rand(B, 3, device=dev) * 0.10 + 0.06) * D
    zz = torch.arange(D, device=dev).view(1, D, 1, 1)
    yy = torch.arange(H, device=dev).view(1, 1, H, 1)
    xx = torch.arange(W, device=dev).view(1, 1, 1, W)
    ell = (((zz - c[:, 0].view(B, 1, 1, 1)) / rad[:, 0].view(B, 1, 1, 1)) ** 2
           + ((yy - c[:, 1].view(B, 1, 1, 1)) / rad[:, 1].view(B, 1, 1, 1)) ** 2
           + ((xx - c[:, 2].view(B, 1, 1, 1)) / rad[:, 2].view(B, 1, 1, 1)) ** 2) <= 1.0
    keep = 1.0 - (ell.float() * do)                                 # zero ellipsoid where do
    return x * keep.unsqueeze(1)


def augment(x, cfg):
    """Raw-intensity augmentation BEFORE MIND: geometry (d2) + intensity (bias/gamma) + resection (d3)."""
    g = {"rot_deg": cfg["rot"], "elastic": cfg["elastic"]}
    x = le._geom_augment(x, g)
    x = le._intensity_augment(x)
    return _resection(x, cfg["resect"])


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.enc = Encoder(6, cfg["dim"], cfg["width"])
        self.dil = cfg["dil"]
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode(self, x):                       # x: raw (B,1,D,H,W) -> L2 embedding
        m = rk._mind(x, self.dil)              # (B,6,D,H,W) modality-invariant structure
        return F.normalize(self.enc(m), dim=1)


def build(pairs, index, grid, loader, device=None):
    cfg = _cfg()
    device = torch.device(device or le.pick_device())
    print(f"[mind] device={device} pairs={len(pairs)} cfg={cfg}")

    q = le._load_stack(pairs, index, grid, loader, "query_id", device)   # (N,1,D,H,W)
    t = le._load_stack(pairs, index, grid, loader, "target_id", device)
    N = len(pairs)

    model = Model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
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
            zq = model.encode(augment(q[idx], cfg))      # independent draws -> q and t differ
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
            print(f"[mind] epoch {epoch:03d} loss={total / max(seen,1):.4f} ({time.time()-t0:.0f}s)")
    model.eval()

    @torch.no_grad()
    def embed(vol_np):
        x = torch.from_numpy(np.ascontiguousarray(vol_np))[None, None].float().to(device)
        if cfg["tta"] <= 1:
            return model.encode(x)[0].cpu().numpy().astype(np.float32)
        # test-time augmentation: average embeddings over mild geometric views, renormalise
        acc = model.encode(x)[0]
        for _ in range(cfg["tta"] - 1):
            xa = le._geom_augment(x, {"rot_deg": cfg["rot"] * 0.5, "elastic": cfg["elastic"] * 0.5})
            acc = acc + model.encode(xa)[0]
        z = F.normalize(acc, dim=0)
        return z.cpu().numpy().astype(np.float32)

    embed.model = model
    return embed
