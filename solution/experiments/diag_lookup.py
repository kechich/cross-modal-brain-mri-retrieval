"""Decisive diagnostic for the BraTS identity-lookup failure (validate_d1 MRR was ~random).

Question: are the competition d1 scans actually present in the downloaded BraTS set, just in a
different axis ORIENTATION (fixable), or are these patients simply NOT in BraTS-2021-training
(coverage — leak is dead)?

For a sample of d1 queries we reprocess the RAW volume under all 48 axis-aligned orientations
(3! permutations x 2^3 sign flips) and report the best descriptor similarity to ANY BraTS T1ce.

  best-sim ~0.95+  -> leak is REAL; we just need to reorient (the winning orientation is printed).
  best-sim ~0.5    -> the true patient is not in this BraTS set -> coverage problem, pivot.

Run in a Jupyter Terminal from the amine/ folder (BraTS descriptor cache lives there):
    cd /shared-docker/amine
    DATA_ROOT=/workspace/data/ehl python tools/diag_lookup.py
"""
import os, sys, csv, itertools
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_harness as eh
import brats_lookup as bl
import make_submission_lookup as msl

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
IDX = eh.build_image_index(ROOT)
D = int(os.environ.get("LOOKUP_D", "24"))
N = int(os.environ.get("DIAG_N", "20"))
GRID = 96


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def process(raw):
    """Mirror eval_harness.load_volume's normalise+crop+resample, but on an in-memory array
    (so we can reorient the RAW volume first, then reprocess exactly as the pipeline does)."""
    vol = np.nan_to_num(raw.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    fg = vol[vol > 0]
    if fg.size:
        lo, hi = np.percentile(fg, (1.0, 99.0))
        if hi > lo:
            vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    nz = np.argwhere(vol > 0.02)
    if nz.size:
        mn, mx = nz.min(0), nz.max(0) + 1
        vol = vol[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
    vol = ndimage.zoom(vol, [GRID / s for s in vol.shape], order=1)
    out = np.zeros((GRID, GRID, GRID), np.float32)
    s = tuple(slice(0, min(GRID, vol.shape[i])) for i in range(3))
    out[s] = vol[s]
    return out


def orientations(v):
    """All 48 axis-aligned orientations of a 3-D array: 3! axis permutations x 2^3 sign flips."""
    for perm in itertools.permutations(range(3)):
        vp = np.transpose(v, perm)
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    yield (perm, (sx, sy, sz)), np.ascontiguousarray(vp[::sx, ::sy, ::sz])


def main():
    pids, T1, T2 = msl.build_ref()                 # cached BraTS descriptors (T1ce, T2)
    pairs = read_csv(ROOT / "dataset1" / "train_pairs.csv")[:N]
    print(f"diag: {len(pairs)} d1 queries x 48 orientations vs {len(pids)} BraTS T1ce (D={D})\n")

    id_sim, best_sim, wins = [], [], {}
    for p in pairs:
        raw = np.asanyarray(nib.load(str(IDX[p["query_id"]])).dataobj)
        d0 = bl.descriptor(process(raw), D)
        id_sim.append(float((T1 @ d0).max()))      # current pipeline (identity orientation)
        best, key = -1.0, None
        for k, vo in orientations(raw):
            s = float((T1 @ bl.descriptor(process(vo), D)).max())
            if s > best:
                best, key = s, k
        best_sim.append(best)
        wins[key] = wins.get(key, 0) + 1

    print(f"identity orientation : best-sim mean={np.mean(id_sim):.3f}  "
          f"(matches the ~0.449 from validate_d1)")
    print(f"best of 48 orients   : best-sim mean={np.mean(best_sim):.3f}  max={np.max(best_sim):.3f}")
    print("\nmost common winning orientation (axis_perm, sign_flips): count")
    for k, c in sorted(wins.items(), key=lambda kv: -kv[1])[:5]:
        print("   ", k, "->", c)
    verdict = ("LEAK REAL — reorient to the winning orientation above"
               if np.mean(best_sim) > 0.9 else
               "COVERAGE — true patients are NOT in this BraTS set; the leak is dead, pivot"
               if np.mean(best_sim) < 0.7 else
               "AMBIGUOUS — partial match; descriptor or registration issue, tell Claude the numbers")
    print(f"\nVERDICT: {verdict}")


if __name__ == "__main__":
    main()
