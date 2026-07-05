"""
Per-dataset submission builder (Track 1: no training, no external data).

Exploits what the data investigation revealed, one strategy per difficulty level:
  * dataset1 (aligned BraTS) -> rank by ALIGNED cross-modal similarity. Query/target share
    the same voxel grid, so edge maps line up across modality: gradient-magnitude cosine
    (+ a mutual-information blend). Measured ~0.75 MRR on labelled d1 pairs, zero training.
  * dataset2 (deformed BraTS) -> not mutually aligned, so use the modality- & deformation-
    invariant anatomical fingerprint (sorted tissue volumes + shape spreads).
  * dataset3 (different clinical set) -> original array SHAPE is a strong partial fingerprint
    (pins ~half of queries to a single gallery image); rank by shape match, fingerprint as
    tiebreaker.

Writes submission.csv (377 rows). Run on the box:  DATA_ROOT=/workspace/data/ehl python make_submission.py
"""
import os, csv, time
from pathlib import Path
import numpy as np
import nibabel as nib

import eval_harness as eh   # reuse build_image_index, load_volume, emb_fingerprint

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
OUT = Path(os.environ.get("OUT", "submission.csv"))
GRID = int(os.environ.get("GRID", "96"))
USE_MI = os.environ.get("USE_MI", "1") == "1"

IDX = eh.build_image_index(ROOT)
print(f"DATA_ROOT={ROOT}  indexed={len(IDX)}  grid={GRID}  use_mi={USE_MI}")

# (dataset, split, strategy)
SETS = [
    ("dataset1", "val", "aligned"), ("dataset1", "test", "aligned"),
    ("dataset2", "val", "invariant"), ("dataset2", "test", "invariant"),
    ("dataset3", "val", "shape"), ("dataset3", "test", "shape"),
]

_vol_cache = {}


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def vol(image_id):
    if image_id not in _vol_cache:
        _vol_cache[image_id] = eh.load_volume(IDX[image_id], GRID)
    return _vol_cache[image_id]


def grad_unit(v):
    gz, gy, gx = np.gradient(v)
    m = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2).ravel().astype(np.float32)
    n = np.linalg.norm(m)
    return m / n if n > 0 else m


def orig_shape(image_id):
    return np.array(nib.load(str(IDX[image_id])).shape[:3], np.float32)


def mutual_information(a, b, bins=32):
    m = (a > 0.02) & (b > 0.02)
    if m.sum() < 50:
        return 0.0
    h, _, _ = np.histogram2d(a[m], b[m], bins=bins, range=[[0, 1], [0, 1]])
    p = h / h.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())


def rank_aligned(qids, gids):
    Gq = np.stack([grad_unit(vol(i)) for i in qids])
    Gg = np.stack([grad_unit(vol(i)) for i in gids])
    scores = Gq @ Gg.T                                   # gradient cosine
    if USE_MI:
        mi = np.array([[mutual_information(vol(q), vol(g)) for g in gids] for q in qids])
        mi = (mi - mi.min()) / (np.ptp(mi) + 1e-9)
        scores = scores + 0.3 * mi
    return scores


def rank_invariant(qids, gids):
    Fq = np.stack([eh.emb_fingerprint(vol(i)) for i in qids])
    Fg = np.stack([eh.emb_fingerprint(vol(i)) for i in gids])
    return Fq @ Fg.T


def rank_shape(qids, gids):
    Sq = np.stack([orig_shape(i) for i in qids])
    Sg = np.stack([orig_shape(i) for i in gids])
    dist = np.linalg.norm(Sq[:, None, :] - Sg[None, :, :], axis=2)   # (Nq,Ng)
    Fq = np.stack([eh.emb_fingerprint(vol(i)) for i in qids])
    Fg = np.stack([eh.emb_fingerprint(vol(i)) for i in gids])
    fp = Fq @ Fg.T
    return -1000.0 * dist + fp        # shape dominates, fingerprint breaks ties


RANKERS = {"aligned": rank_aligned, "invariant": rank_invariant, "shape": rank_shape}


def main():
    rows = []
    for ds, split, strat in SETS:
        t0 = time.time()
        qids = [r["query_id"] for r in read_csv(ROOT / ds / f"{split}_queries.csv")]
        gids = [r["target_id"] for r in read_csv(ROOT / ds / f"{split}_gallery.csv")]
        scores = RANKERS[strat](qids, gids)
        for i, q in enumerate(qids):
            order = np.argsort(-scores[i])
            rows.append({"query_id": q, "target_id_ranking": " ".join(gids[j] for j in order)})
        print(f"  {ds}/{split:4s} [{strat:9s}] {len(qids)}q x {len(gids)}g  ({time.time()-t0:.0f}s)")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
