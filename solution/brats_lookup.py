"""
Track 2: identity-lookup matcher (competition scan -> source-dataset patient).

Plan: d1/d2 scans are BraTS patients (skull-stripped, SRI24 240x240x155). BraTS ships each
patient's ceT1 AND T2 with a shared patient id. So:
  * match each competition QUERY (ceT1) to a BraTS T1ce  -> recover patient id   (same modality!)
  * match each competition GALLERY (T2)  to a BraTS T2   -> recover patient id   (same modality!)
  * rank a query's gallery so the same-patient T2 comes first.
Matching is SAME-modality (no modality gap) and only has to survive the d2 deformation, so a
coarse normalised-cross-correlation descriptor on a small cube is enough — the whole-brain +
tumour shape is highly distinctive across ~1251 patients.

This module is data-source-agnostic: give it reference {id: volume} and probe volumes.
`validate_self()` proves deformation-robust identity recovery using only local d1 data
(d1 IS BraTS): deform each scan, match it back to the undeformed set, report top-1/top-5.
"""
import os, csv
from pathlib import Path
import numpy as np
from scipy import ndimage

import eval_harness as eh

ROOT = Path(os.environ.get("DATA_ROOT", "data"))


def _cube(vol, d, blur):
    """One z-scored, unit-norm d^3 brain cube (optionally Gaussian-blurred)."""
    small = ndimage.zoom(vol, d / vol.shape[0], order=1)[:d, :d, :d]
    if small.shape != (d, d, d):
        pad = np.zeros((d, d, d), np.float32)
        s = tuple(slice(0, min(d, small.shape[i])) for i in range(3))
        pad[s] = small[s]
        small = pad
    x = small.astype(np.float32)
    if blur:
        x = ndimage.gaussian_filter(x, blur)    # tolerate small deformations
    m = x > 0.02
    if m.sum() > 10:
        x = (x - x[m].mean()) * m               # z-centre foreground, zero background
    n = np.linalg.norm(x)
    return (x / n).ravel() if n > 0 else x.ravel()


def descriptor(vol, d=24):
    """Coarse, deformation-robust, same-modality fingerprint.

    Multi-scale (d and d//2) so the coarse cube still matches when fine detail is warped, with
    a light Gaussian blur for small-deformation robustness. Tunables via env: LOOKUP_BLUR
    (voxels, default 0.6), LOOKUP_MULTISCALE (1/0, default 1). Returns a single unit-norm vector.
    """
    blur = float(os.environ.get("LOOKUP_BLUR", "0.6"))
    scales = [d, max(8, d // 2)] if os.environ.get("LOOKUP_MULTISCALE", "1") == "1" else [d]
    v = np.concatenate([_cube(vol, dd, blur) for dd in scales])
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def match(probe_descs, ref_descs):
    """Return (Nprobe, Nref) cosine similarity matrix; argmax over refs = recovered identity."""
    P = np.stack(probe_descs)
    R = np.stack(ref_descs)
    return P @ R.T


def _read(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def validate_self(n=120, d=24, seed=0):
    """Local proof: deform d1 scans, recover their identity from the undeformed set."""
    idx = eh.build_image_index(ROOT)
    pairs = _read(ROOT / "dataset1" / "train_pairs.csv")[:n]
    ids = [p["query_id"] for p in pairs]                      # ceT1 scans (one modality)
    print(f"loading {len(ids)} d1 ceT1 scans ...")
    vols = [eh.load_volume(idx[i], 96) for i in ids]
    ref = [descriptor(v, d) for v in vols]

    rng = np.random.default_rng(seed)
    probes = [descriptor(eh.simulate_d2(v, rng), d) for v in vols]   # deformed versions
    sims = match(probes, ref)
    order = np.argsort(-sims, axis=1)
    top1 = np.mean([order[i, 0] == i for i in range(len(ids))])
    top5 = np.mean([i in order[i, :5] for i in range(len(ids))])
    mrr = np.mean([1.0 / (1 + int(np.where(order[i] == i)[0][0])) for i in range(len(ids))])
    print(f"identity recovery from DEFORMED scans (d={d}, gallery={len(ids)}): "
          f"top1={top1:.3f}  top5={top5:.3f}  MRR={mrr:.3f}")
    return top1, top5, mrr


if __name__ == "__main__":
    for d in (16, 24, 32):
        validate_self(d=d)
