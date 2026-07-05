"""Split a full submission into per-dataset files to read each dataset's TRUE MRR from Kaggle.

Submit one of the split files (it omits the other datasets, which then score 0); the displayed
leaderboard score x3 = that dataset's MRR. This reveals which of d1/d2/d3 is the real bottleneck
so we stop optimizing blind against the (misleading) offline proxy.

    DATA_ROOT=/workspace/data/ehl python split_submission.py submission_content.csv
-> writes submission_d1.csv (40+100 rows), submission_d2.csv (40+100), submission_d3.csv (20+77)
"""
import os, sys, csv
from pathlib import Path

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
SRC = sys.argv[1] if len(sys.argv) > 1 else "submission_content.csv"


def dataset_qids(ds):
    ids = set()
    for split in ("val", "test"):
        with open(ROOT / ds / f"{split}_queries.csv", newline="") as f:
            for r in csv.DictReader(f):
                ids.add(r["query_id"])
    return ids


def main():
    with open(SRC, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"{SRC}: {len(rows)} rows")
    for ds in ("dataset1", "dataset2", "dataset3"):
        ids = dataset_qids(ds)
        sub = [r for r in rows if r["query_id"] in ids]
        out = f"submission_{ds[-1]}.csv"           # submission_1 / _2 / _3
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
            w.writeheader()
            w.writerows(sub)
        print(f"  {ds}: {len(sub)} rows -> {out}   (submit it; displayed score x3 = {ds} MRR)")


if __name__ == "__main__":
    main()
