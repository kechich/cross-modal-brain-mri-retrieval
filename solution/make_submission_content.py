"""Best content-only submission (no external data) — the guaranteed floor.

Per-dataset strategy, each matched to what the data allows:
  * dataset1 (aligned, shared grid)  -> DENSE MIND-field cosine across the common grid
                                        (rk.rank_dense_mind). ~0.71 MRR on a 100-gallery — best
                                        content method; keeps spatial correspondence the global
                                        rankers throw away.
  * dataset2 (independently deformed) -> MIND only (modality- & deformation-tolerant); the
                                        alignment-based scores are useless once the grid breaks.
  * dataset3 (pre/intra-op, varied shape) -> the original-array SHAPE prior (pins ~half the
                                        queries), MIND as tiebreaker. Reuses make_submission.rank_shape.

GPU rankers run on the MI300X. Run on the box:
    DATA_ROOT=/workspace/data/ehl OUT=submission.csv python make_submission_content.py
"""
import os, csv, time
from pathlib import Path
import numpy as np

import eval_harness as eh
import rankers as rk
import make_submission as ms      # reuse rank_shape for d3

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
OUT = Path(os.environ.get("OUT", "submission.csv"))
GRID = int(os.environ.get("GRID", "96"))

IDX = eh.build_image_index(ROOT)
ms.IDX = IDX                                   # rank_shape reads volumes through ms.IDX
DEV = rk.pick_device()
print(f"DATA_ROOT={ROOT} indexed={len(IDX)} grid={GRID} device={DEV}")

_cache = {}


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def load(ids):
    out = []
    for i in ids:
        if i not in _cache:
            _cache[i] = eh.load_volume(IDX[i], GRID)
        out.append(_cache[i])
    return out


def _rn(s):
    lo = s.min(1, keepdims=True)
    hi = s.max(1, keepdims=True)
    return (s - lo) / (hi - lo + 1e-9)


def rank_d1(qids, gids):
    q, g = load(qids), load(gids)
    return rk.rank_dense_mind(q, g, DEV)        # 0.71 vs 0.62 for the global blend


def rank_d2(qids, gids):
    q, g = load(qids), load(gids)
    return rk.rank_mind(q, g, DEV)


D3_MIND_W = float(os.environ.get("D3_MIND_W", "0"))   # >0 adds a global-MIND tiebreak to d3


def rank_d3(qids, gids):
    base = ms.rank_shape(qids, gids)           # shape-dominant + fingerprint tiebreak (the 0.61 d3)
    if D3_MIND_W > 0:                          # opt-in: MIND breaks ties the shape prior leaves
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
        print(f"  {ds}/{split:4s} [{ranker.__name__:8s}] {len(qids)}q x {len(gids)}g "
              f"({time.time() - t0:.0f}s)")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT} (expect 377)")


if __name__ == "__main__":
    main()
