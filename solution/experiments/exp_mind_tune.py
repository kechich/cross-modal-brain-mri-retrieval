"""Tune dense MIND for d1 (it won the first experiment at 0.714). Two main levers:
  * foreground masking — background MIND is all-ones (no structure) and adds a constant
    correlated term to every pair, diluting the real signal. Zeroing it should help.
  * neighbourhood / dilation / resolution — richer structural context.

Run on the box GPU, paste the table:
    cd /shared-docker/amine && DATA_ROOT=/workspace/data/ehl python tools/exp_mind_tune.py
"""
import os, sys, csv, time, itertools
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


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


pairs = read_csv(ROOT / "dataset1" / "train_pairs.csv")[:N]
qi = [p["query_id"] for p in pairs]
gi = [p["target_id"] for p in pairs]
true = np.arange(len(pairs))


def stack(ids, G):
    return torch.from_numpy(np.stack([eh.load_volume(IDX[i], G) for i in ids]).astype("float32")).to(DEV)


def mrr(s):
    return round(eh.mrr_from_scores(s, true), 3)


FACE = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
EDGE = [(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0), (1, 0, 1), (1, 0, -1),
        (-1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)]


def mind_field(x, dilation, offsets, mask_bg):
    """x:(N,1,D,H,W) -> normalised dense MIND vectors (N, C*D*H*W)."""
    feats = []
    for o in offsets:
        s = tuple(d * dilation for d in o)
        xs = torch.roll(x, shifts=s, dims=(2, 3, 4))
        feats.append(F.avg_pool3d((x - xs) ** 2, 3, 1, 1))
    dp = torch.cat(feats, 1)
    var = dp.mean(1, keepdim=True).clamp_min(1e-6)
    m = torch.exp(-dp / var)
    m = m / m.amax(1, keepdim=True).clamp_min(1e-6)
    if mask_bg:
        m = m * (x > 0.05).float()                # zero structure-less background
    return F.normalize(m.reshape(x.shape[0], -1), dim=1)


def score(G, dilation, offsets, mask_bg):
    q = stack(qi, G).unsqueeze(1)
    g = stack(gi, G).unsqueeze(1)
    qf = mind_field(q, dilation, offsets, mask_bg)
    gf = mind_field(g, dilation, offsets, mask_bg)
    return mrr((qf @ gf.t()).cpu().numpy())


def main():
    print(f"dense-MIND tuning on d1: N={len(pairs)}, device={DEV}\n")
    print(f"{'config':46s}  d1-MRR")
    print("-" * 58)
    t0 = time.time()
    trials = [
        ("G96 d2 face6  mask=0 (baseline)", 96, 2, FACE, False),
        ("G96 d2 face6  mask=1", 96, 2, FACE, True),
        ("G96 d1 face6  mask=1", 96, 1, FACE, True),
        ("G96 d3 face6  mask=1", 96, 3, FACE, True),
        ("G96 d2 face+edge18 mask=1", 96, 2, FACE + EDGE, True),
        ("G96 d2 edge12 mask=1", 96, 2, EDGE, True),
        ("G112 d2 face6 mask=1", 112, 2, FACE, True),
        ("G128 d2 face6 mask=1", 128, 2, FACE, True),
        ("G128 d2 face+edge18 mask=1", 128, 2, FACE + EDGE, True),
    ]
    best = None
    for name, G, dil, offs, mb in trials:
        v = score(G, dil, offs, mb)
        print(f"{name:46s}  {v:.3f}")
        if best is None or v > best[1]:
            best = (name, v)
    print(f"\nbest: {best[0]} = {best[1]:.3f}   (prev dense_mind 0.714, blend 0.619)")
    print(f"(total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
