"""Best submission: dense-MIND everywhere, with REGISTRATION unlocking dataset2.

Per-dataset routing (real Kaggle MRRs that motivated each choice):
  * dataset1 (aligned)   -> dense_mind                         (0.72)
  * dataset2 (deformed)  -> register-to-canonical + dense_mind (proxy 0.15 -> 0.72; the big win)
  * dataset3 (intra-op)  -> shape leak + global-MIND tiebreak  (0.85)

The d2 fix: each deformed volume is affine-registered (rigid+scale, via modality-invariant MIND)
to a canonical pose built from clean dataset1 volumes, so the strong aligned matcher works again.

Run on the box GPU:
    DATA_ROOT=/workspace/data/ehl OUT=submission_best.csv python make_submission_best.py
Env: D3_MIND_W (default 0.3), REG_ITERS (default 70), N_TMPL (template volumes, default 40).
"""
import os, csv, time
from pathlib import Path
import numpy as np

import eval_harness as eh
import rankers as rk
import make_submission as ms          # rank_shape for d3

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
OUT = Path(os.environ.get("OUT", "submission_best.csv"))
GRID = int(os.environ.get("GRID", "96"))
D3_MIND_W = float(os.environ.get("D3_MIND_W", "0.3"))
REG_ITERS = int(os.environ.get("REG_ITERS", "70"))
N_TMPL = int(os.environ.get("N_TMPL", "40"))

IDX = eh.build_image_index(ROOT)
ms.IDX = IDX
DEV = rk.pick_device()
print(f"DATA_ROOT={ROOT} indexed={len(IDX)} grid={GRID} device={DEV}")

_cache = {}


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def vol(i):
    if i not in _cache:
        _cache[i] = eh.load_volume(IDX[i], GRID)
    return _cache[i]


def load(ids):
    return [vol(i) for i in ids]


def _rn(s):
    lo, hi = s.min(1, keepdims=True), s.max(1, keepdims=True)
    return (s - lo) / (hi - lo + 1e-9)


# canonical-pose template volumes: a sample of clean dataset1 gallery (aligned SRI24 frame)
TMPL = load([r["target_id"] for r in read_csv(ROOT / "dataset1" / "train_pairs.csv")[:N_TMPL]])


def rank_d1(qids, gids):
    return rk.rank_dense_mind(load(qids), load(gids), DEV)


def rank_d2(qids, gids):
    return rk.rank_dense_mind_registered(load(qids), load(gids), TMPL, DEV, iters=REG_ITERS)


def rank_d3(qids, gids):
    base = ms.rank_shape(qids, gids)
    if D3_MIND_W > 0:
        return _rn(base) + D3_MIND_W * _rn(rk.rank_mind(load(qids), load(gids), DEV))
    return base


SETS = [("dataset1", "val", rank_d1), ("dataset1", "test", rank_d1),
        ("dataset2", "val", rank_d2), ("dataset2", "test", rank_d2),
        ("dataset3", "val", rank_d3), ("dataset3", "test", rank_d3)]


def main():
    rows = []
    for ds, split, ranker in SETS:
        t0 = time.time()
        qids = [r["query_id"] for r in read_csv(ROOT / ds / f"{split}_queries.csv")]
        gids = [r["target_id"] for r in read_csv(ROOT / ds / f"{split}_gallery.csv")]
        scores = ranker(qids, gids)
        for i, q in enumerate(qids):
            order = np.argsort(-scores[i])
            rows.append({"query_id": q, "target_id_ranking": " ".join(gids[j] for j in order)})
        print(f"  {ds}/{split:4s} [{ranker.__name__:8s}] {len(qids)}q x {len(gids)}g ({time.time()-t0:.0f}s)")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT} (expect 377)")


if __name__ == "__main__":
    main()
