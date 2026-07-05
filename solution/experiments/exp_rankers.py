"""Experiment: stronger d1 rankers that EXPLOIT the common-grid alignment.

The current rankers throw away spatial correspondence (global NMI = one histogram; global-pooled
MIND = one vector). On dataset1 the query and target are voxel-aligned, so dense / local matching
of modality-invariant structure should be far more discriminative. This script scores candidates
on the 100 labelled d1 pairs (gallery=100, the real test size) and prints an MRR table.

Run on the box GPU, paste the table back to Claude:
    cd /shared-docker/amine && DATA_ROOT=/workspace/data/ehl python tools/exp_rankers.py
Env: EXP_N (pairs, default 100), EXP_GRID (default 96).
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
N = int(os.environ.get("EXP_N", "100"))
G = int(os.environ.get("EXP_GRID", "96"))


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


pairs = read_csv(ROOT / "dataset1" / "train_pairs.csv")[:N]
qi = [p["query_id"] for p in pairs]
gi = [p["target_id"] for p in pairs]
true = np.arange(len(pairs))


def stack(ids):
    return torch.from_numpy(np.stack([eh.load_volume(IDX[i], G) for i in ids]).astype("float32")).to(DEV)


def mrr(s):
    return round(eh.mrr_from_scores(s, true), 3)


def rn(s):
    lo = s.min(1, keepdims=True)
    hi = s.max(1, keepdims=True)
    return (s - lo) / (hi - lo + 1e-9)


# ---------------- dense (alignment-exploiting) candidates ----------------
def dense_mind(x):                       # MIND field cosine across the aligned grid
    m = rk._mind(x.unsqueeze(1))
    return F.normalize(m.reshape(x.shape[0], -1), dim=1)


def dense_gradorient(x):                 # gradient-ORIENTATION field cosine (modality-robust)
    gz, gy, gx = torch.gradient(x, dim=(1, 2, 3))
    mag = torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2) + 1e-6
    keep = (mag > mag.mean()).unsqueeze(1)
    o = torch.stack([gz / mag, gy / mag, gx / mag], 1) * keep
    return F.normalize(o.reshape(x.shape[0], -1), dim=1)


def tensor_nmi(qf, gf, bins):            # (Nq,V),(Ng,V) -> (Nq,Ng) NMI
    qb = (qf.clamp(0, 1) * (bins - 1)).long()
    gb = (gf.clamp(0, 1) * (bins - 1)).long()
    Nq, Ng, V = qb.shape[0], gb.shape[0], qb.shape[1]
    ones = torch.ones(Ng, V, device=qf.device)
    out = torch.zeros(Nq, Ng, device=qf.device)
    for i in range(Nq):
        comb = qb[i].unsqueeze(0) * bins + gb
        joint = torch.zeros(Ng, bins * bins, device=qf.device)
        joint.scatter_add_(1, comb, ones)
        joint = joint.reshape(Ng, bins, bins)
        pij = joint / joint.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        pi, pj = pij.sum(2), pij.sum(1)
        ent = lambda p: -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)
        hij = -(pij.clamp_min(1e-12) * pij.clamp_min(1e-12).log()).sum(dim=(1, 2))
        out[i] = (ent(pi) + ent(pj)) / hij.clamp_min(1e-12)
    return out


def block_nmi(q, g, K=4, bins=24):       # LOCAL MI: average NMI over a K^3 grid of blocks
    N_, D = q.shape[0], q.shape[1]
    b = D // K
    acc = torch.zeros(N_, g.shape[0], device=q.device)
    for zi in range(K):
        for yi in range(K):
            for xi in range(K):
                sl = (slice(zi * b, (zi + 1) * b), slice(yi * b, (yi + 1) * b), slice(xi * b, (xi + 1) * b))
                qs = q[:, sl[0], sl[1], sl[2]].reshape(N_, -1)
                gs = g[:, sl[0], sl[1], sl[2]].reshape(g.shape[0], -1)
                acc += tensor_nmi(qs, gs, bins)
    return (acc / (K ** 3)).cpu().numpy()


def main():
    print(f"d1 experiment: N={len(pairs)} gallery, grid={G}, device={DEV}\n")
    q = stack(qi)
    g = stack(gi)
    qnp = [v.cpu().numpy() for v in q]     # the existing rankers take numpy-volume lists
    gnp = [v.cpu().numpy() for v in g]
    res = {}

    t = time.time()
    res["global_nmi"] = mrr(rk.rank_nmi(qnp, gnp, DEV))
    res["global_gradcos"] = mrr(rk.rank_gradcos(qnp, gnp, DEV))
    res["global_mind"] = mrr(rk.rank_mind(qnp, gnp, DEV))

    dm = (dense_mind(q) @ dense_mind(g).t()).cpu().numpy()
    do = (dense_gradorient(q) @ dense_gradorient(g).t()).cpu().numpy()
    res["dense_mind"] = mrr(dm)
    res["dense_gradorient"] = mrr(do)

    bn4 = block_nmi(q, g, K=4)
    bn6 = block_nmi(q, g, K=6)
    res["block_nmi_K4"] = mrr(bn4)
    res["block_nmi_K6"] = mrr(bn6)

    # blends of the strongest aligned signals
    res["dense_mind+gradorient"] = mrr(rn(dm) + rn(do))
    res["dense_mind+blockNMI6"] = mrr(rn(dm) + rn(bn6))
    res["dense_all"] = mrr(rn(dm) + rn(do) + rn(bn6))

    print(f"(computed in {time.time()-t:.0f}s)\n")
    print(f"{'method':28s}  d1-MRR")
    print("-" * 40)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"{k:28s}  {v:.3f}")
    print(f"\nbaseline to beat: current best blend ~0.619")


if __name__ == "__main__":
    main()
