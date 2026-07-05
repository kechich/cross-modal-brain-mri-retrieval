"""Test the d2 fix: register deformed images to a canonical pose, THEN dense_mind.

d2 = d1 + independent rigid+elastic deformation. dense_mind scores 0.72 on aligned d1 but needs a
shared frame. If we affine-register each d2 image back toward a canonical template (using
modality-invariant MIND so it works cross-modal), the strong aligned matcher should work on d2.

Compares, on the d2 proxy (simulate_d2 on held-out d1 pairs):
  * global_mind (deformed)        -- the current d2 method
  * dense_mind (deformed, no reg) -- shows the alignment really is the problem
  * dense_mind (registered)       -- the proposed fix

Run on the box GPU:  cd /shared-docker/amine && DATA_ROOT=/workspace/data/ehl python tools/exp_d2_register.py
Env: EXP_N (pairs, default 60), REG_ITERS (default 70).
"""
import os, sys, csv, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_harness as eh
import rankers as rk

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
IDX = eh.build_image_index(ROOT)
DEV = rk.pick_device()
N = int(os.environ.get("EXP_N", "60"))
ITERS = int(os.environ.get("REG_ITERS", "70"))
G = 96


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


pairs = read_csv(ROOT / "dataset1" / "train_pairs.csv")[:N]
qi = [p["query_id"] for p in pairs]
gi = [p["target_id"] for p in pairs]
true = np.arange(len(pairs))


def stack(ids):
    return torch.from_numpy(np.stack([eh.load_volume(IDX[i], G) for i in ids]).astype("float32")).to(DEV)


def mind(x, dil=2):
    return rk._mind(x, dil)


def rot_from_angles(a):
    B, dev = a.shape[0], a.device
    cz, sz = torch.cos(a[:, 0]), torch.sin(a[:, 0])
    cy, sy = torch.cos(a[:, 1]), torch.sin(a[:, 1])
    cx, sx = torch.cos(a[:, 2]), torch.sin(a[:, 2])
    z, o = torch.zeros(B, device=dev), torch.ones(B, device=dev)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(B, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(B, 3, 3)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(B, 3, 3)
    return Rz @ Ry @ Rx


def dense_rank(q, g, dil=2):
    qf = F.normalize(mind(q, dil).reshape(q.shape[0], -1), dim=1)
    gf = F.normalize(mind(g, dil).reshape(g.shape[0], -1), dim=1)
    return (qf @ gf.t()).cpu().numpy()


def register(vol, tmpl_mind, iters=ITERS, lr=0.05, dil=2):
    """Affine-register each volume (B,1,D,H,W) toward a canonical MIND template by maximising
    MIND cosine. Rigid + isotropic scale (init identity) to avoid degenerate collapse."""
    B = vol.shape[0]
    ang = torch.zeros(B, 3, device=DEV, requires_grad=True)
    tr = torch.zeros(B, 3, device=DEV, requires_grad=True)
    ls = torch.zeros(B, 1, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([ang, tr, ls], lr=lr)
    Tm = F.normalize(tmpl_mind.reshape(1, -1), dim=1)
    for _ in range(iters):
        R = rot_from_angles(ang) * torch.exp(ls)[:, None, :]
        theta = torch.cat([R, tr[:, :, None]], dim=2)
        grid = F.affine_grid(theta, vol.shape, align_corners=False)
        warped = F.grid_sample(vol, grid, align_corners=False)
        wm = F.normalize(mind(warped, dil).reshape(B, -1), dim=1)
        loss = -(wm * Tm).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        R = rot_from_angles(ang) * torch.exp(ls)[:, None, :]
        theta = torch.cat([R, tr[:, :, None]], dim=2)
        grid = F.affine_grid(theta, vol.shape, align_corners=False)
        return F.grid_sample(vol, grid, align_corners=False).detach()


def main():
    print(f"d2-registration test: N={len(pairs)} gallery, grid={G}, iters={ITERS}, device={DEV}\n")
    qv = stack(qi).unsqueeze(1)
    gv = stack(gi).unsqueeze(1)
    # canonical template = MIND of the mean clean gallery (a fuzzy average pose)
    tmpl = mind(gv.mean(0, keepdim=True))

    rng = np.random.default_rng(0)
    qd = torch.stack([torch.from_numpy(eh.simulate_d2(v[0].cpu().numpy(), rng)) for v in qv])[:, None].to(DEV)
    gd = torch.stack([torch.from_numpy(eh.simulate_d2(v[0].cpu().numpy(), rng)) for v in gv])[:, None].to(DEV)

    def mrr(s):
        return round(eh.mrr_from_scores(s, true), 3)

    qnp = [v[0].cpu().numpy() for v in qd]
    gnp = [v[0].cpu().numpy() for v in gd]
    print(f"{'global_mind   (deformed)':32s} {mrr(rk.rank_mind(qnp, gnp, DEV)):.3f}   <- current d2")
    print(f"{'dense_mind    (deformed,no reg)':32s} {mrr(dense_rank(qd, gd)):.3f}")
    t = time.time()
    qr = register(qd, tmpl)
    gr = register(gd, tmpl)
    print(f"{'dense_mind    (registered)':32s} {mrr(dense_rank(qr, gr)):.3f}   <- proposed "
          f"({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
