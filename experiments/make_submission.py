"""Generate submission CSV from the trained MIND-CLIP embedder.

Run on AMD box:
    DATA_ROOT=/workspace/data/ehl python make_submission.py

Iterates all 6 sets: dataset{1,2,3} x {val, test}.
Writes: submission.csv (377 rows).
"""
import os
import csv
import time
import sys
from pathlib import Path

import numpy as np

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
OUT = Path(os.environ.get("OUT", "submission.csv"))
GRID = int(os.environ.get("GRID", "96"))

SHARED = Path("/shared-docker")
sys.path.insert(0, str(SHARED / "amine"))
sys.path.insert(0, str(SHARED / "mohamed"))

from eval_harness import build_image_index, cached_volume, read_csv
from learned_embedder import build


SETS = [
    ("dataset1", "val"),
    ("dataset1", "test"),
    ("dataset2", "val"),
    ("dataset2", "test"),
    ("dataset3", "val"),
    ("dataset3", "test"),
]


def main():
    print(f"DATA_ROOT={DATA_ROOT}  grid={GRID}")

    index = build_image_index(DATA_ROOT)
    print(f"Indexed {len(index)} images")

    train_pairs = read_csv(DATA_ROOT / "dataset1" / "train_pairs.csv")
    print(f"Training on {len(train_pairs)} pairs...")
    embed_fn = build(train_pairs, index, GRID, cached_volume)

    rows = []
    for ds, split in SETS:
        t0 = time.time()
        qids = [r["query_id"] for r in read_csv(DATA_ROOT / ds / f"{split}_queries.csv")]
        gids = [r["target_id"] for r in read_csv(DATA_ROOT / ds / f"{split}_gallery.csv")]

        q_embs = np.stack([embed_fn(cached_volume(q, index[q], GRID)) for q in qids])
        g_embs = np.stack([embed_fn(cached_volume(g, index[g], GRID)) for g in gids])
        scores = q_embs @ g_embs.T   # (Nq, Ng) cosine

        for i, q in enumerate(qids):
            order = np.argsort(-scores[i])
            rows.append({
                "query_id": q,
                "target_id_ranking": " ".join(gids[j] for j in order),
            })

        print(f"  {ds}/{split:4s}  {len(qids)}q x {len(gids)}g  ({time.time()-t0:.0f}s)")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {OUT}  (expect 377)")


if __name__ == "__main__":
    main()
