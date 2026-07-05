"""
Offline d1/d2/d3 validation harness for the brain-MRI cross-modal retrieval challenge.

WHY THIS EXISTS
---------------
The only labelled data we have is dataset1's 350 aligned train pairs. The real val/test
matches are hidden and we get only 100 Kaggle submissions/day. So we cannot tune against
the leaderboard. This harness builds SYNTHETIC proxies of all three difficulty levels from
the held-out dataset1 pairs, so any retrieval method can be scored offline in minutes:

  * d1-proxy : real ceT1 query vs real T2 gallery, untouched  -> tests the MODALITY GAP only.
  * d2-proxy : + independent random rigid + elastic deformation on each side
               -> tests deformation invariance (mimics real dataset2).
  * d3-proxy : + synthetic resection (cavity) + bias field + intensity/gamma shift
               -> tests robustness to structural change + domain shift (mimics dataset3).

The modality gap is REAL in every level (query is a true ceT1, gallery a true T2); the
simulators only add geometry/domain perturbation on top. Macro score = mean of the three.

An "embedder" is any function  volume(np.float32, HxWxD) -> 1-D L2-normalised vector.
Plug new methods into EMBEDDERS and rerun. No GPU needed for the reference embedders
(numpy + scipy + nibabel only); a trained model just becomes another embedder.

USAGE
-----
    DATA_ROOT=/workspace/data/ehl python eval_harness.py
    # options via env: N_VAL (held-out pairs), GRID (cube size), SEED, OUT (results md)

Writes a results table to stdout and to eval_results.md.
"""
import os
import csv
import json
import time
from pathlib import Path

import numpy as np

# scipy is the only heavy dep for the simulators / reference embedders.
from scipy import ndimage

try:
    import nibabel as nib
except Exception:  # pragma: no cover - only needed for the real run
    nib = None


# --------------------------------------------------------------------------- config
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
N_VAL = int(os.environ.get("N_VAL", "60"))          # held-out dataset1 pairs to score on
GRID = int(os.environ.get("GRID", "96"))            # volumes are resampled to GRID^3
SEED = int(os.environ.get("SEED", "20260627"))
OUT = Path(os.environ.get("OUT", "eval_results.md"))
CACHE = Path(os.environ.get("CACHE", ".vol_cache"))  # normalised-volume cache (.npy)


# --------------------------------------------------------------------------- io / index
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_image_index(root: Path):
    """Map every image ID (filename stem) to its real path. Robust to .nii / .nii.gz."""
    index = {}
    for p in root.glob("**/*.nii*"):
        name = p.name
        stem = name[:-7] if name.endswith(".nii.gz") else name[:-4]
        index[stem] = p
    return index


def load_volume(path: Path, grid: int) -> np.ndarray:
    """Load a NIfTI -> brain-cropped, intensity-normalised, resampled to grid^3 float32."""
    vol = np.asanyarray(nib.load(str(path)).dataobj).astype(np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)
    # robust intensity normalisation to [0,1] using foreground percentiles
    fg = vol[vol > 0]
    if fg.size:
        lo, hi = np.percentile(fg, (1.0, 99.0))
        if hi > lo:
            vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
    # crop to nonzero brain bounding box
    nz = np.argwhere(vol > 0.02)
    if nz.size:
        mn, mx = nz.min(0), nz.max(0) + 1
        vol = vol[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
    # resample to a fixed cube so geometry is comparable across scans
    factors = [grid / s for s in vol.shape]
    vol = ndimage.zoom(vol, factors, order=1)
    # zoom can over/undershoot by 1 voxel; pad/crop to exact grid
    out = np.zeros((grid, grid, grid), np.float32)
    s = tuple(slice(0, min(grid, vol.shape[i])) for i in range(3))
    out[s] = vol[s]
    return out


def cached_volume(image_id: str, path: Path, grid: int) -> np.ndarray:
    CACHE.mkdir(parents=True, exist_ok=True)
    cp = CACHE / f"{image_id}_{grid}.npy"
    if cp.exists():
        return np.load(cp)
    v = load_volume(path, grid)
    np.save(cp, v)
    return v


# --------------------------------------------------------------------------- simulators
def _elastic(vol, rng, alpha, sigma):
    """Random smooth displacement field warp (mimics non-linear deformation)."""
    shape = vol.shape
    disp = [ndimage.gaussian_filter((rng.random(shape) * 2 - 1), sigma) * alpha
            for _ in range(3)]
    grid = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    coords = [g + d for g, d in zip(grid, disp)]
    return ndimage.map_coordinates(vol, coords, order=1, mode="constant")


def simulate_d2(vol, rng):
    """Independent rigid (rotation+shift) + elastic warp. Mimics dataset2."""
    out = vol
    # small rotations about each axis
    for axes in [(0, 1), (0, 2), (1, 2)]:
        ang = rng.uniform(-15, 15)
        out = ndimage.rotate(out, ang, axes=axes, reshape=False, order=1, mode="constant")
    # translation
    shift = rng.uniform(-6, 6, size=3)
    out = ndimage.shift(out, shift, order=1, mode="constant")
    # non-linear deformation
    out = _elastic(out, rng, alpha=GRID * 0.08, sigma=GRID * 0.10)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def simulate_d3(vol, rng):
    """d2-style geometry + synthetic resection cavity + bias field + intensity shift.

    Mimics dataset3: pre/post-surgery tissue change AND a different scanner/domain."""
    out = simulate_d2(vol, rng)
    # bias field: smooth low-frequency multiplicative gain (scanner inhomogeneity)
    bias = ndimage.gaussian_filter(rng.random(out.shape), GRID * 0.25)
    bias = 0.7 + 0.6 * (bias - bias.min()) / (np.ptp(bias) + 1e-6)
    out = out * bias
    # domain intensity shift: random gamma
    out = np.clip(out, 0, None)
    out = out ** rng.uniform(0.6, 1.6)
    # resection: zero out an ellipsoid somewhere inside the brain
    brain = np.argwhere(out > 0.05)
    if brain.size:
        c = brain[rng.integers(len(brain))]
        rad = rng.uniform(GRID * 0.08, GRID * 0.16, size=3)
        zz, yy, xx = np.ogrid[:out.shape[0], :out.shape[1], :out.shape[2]]
        ell = (((zz - c[0]) / rad[0]) ** 2 + ((yy - c[1]) / rad[1]) ** 2
               + ((xx - c[2]) / rad[2]) ** 2) <= 1.0
        out[ell] = 0.0
    m = out.max()
    if m > 0:
        out = out / m
    return out.astype(np.float32)


SIMULATORS = {"d1": lambda v, rng: v, "d2": simulate_d2, "d3": simulate_d3}


# --------------------------------------------------------------------------- embedders
def _l2(x):
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def emb_intensity(vol):
    """Downsampled raw intensities. Modality- AND alignment-sensitive -> the weak floor."""
    small = ndimage.zoom(vol, 16 / vol.shape[0], order=1)[:16, :16, :16]
    return _l2(small.ravel())


def emb_edges(vol):
    """Histogram of gradient magnitudes + a coarse 3-D gradient-orientation grid.

    Edges sit at the same anatomical boundaries in both modalities (contrast may invert
    but structure is shared) -> partially modality- and deformation-robust."""
    gz, gy, gx = np.gradient(vol)
    mag = np.sqrt(gz ** 2 + gy ** 2 + gx ** 2)
    hist, _ = np.histogram(mag[mag > 0], bins=24, range=(0, mag.max() + 1e-6), density=True)
    # coarse spatial energy grid (where are the strong edges) at 6^3
    block = ndimage.zoom(mag, 6 / mag.shape[0], order=1)[:6, :6, :6].ravel()
    return _l2(np.concatenate([hist, block]))


def emb_fingerprint(vol):
    """Crude, modality-INVARIANT anatomical fingerprint (a cheap SynthSeg stand-in).

    Cluster brain voxels into 3 intensity tissues, then SORT clusters by volume so the
    descriptor does not depend on which modality makes which tissue bright (T1<->T2 invert).
    Features per sorted cluster: volume fraction + spatial spread (std on each axis).
    Volumes/topology are largely preserved under deformation, so this should hold up on
    d2/d3 where intensity/alignment methods collapse."""
    mask = vol > 0.05
    vals = vol[mask]
    if vals.size < 10:
        return _l2(np.zeros(12, np.float32))
    # 3 intensity bins via terciles of foreground
    t1, t2 = np.percentile(vals, (33.3, 66.6))
    labels = np.zeros(vol.shape, np.int8)
    labels[mask & (vol <= t1)] = 1
    labels[mask & (vol > t1) & (vol <= t2)] = 2
    labels[mask & (vol > t2)] = 3
    feats = []
    total = mask.sum() + 1e-6
    for k in (1, 2, 3):
        coords = np.argwhere(labels == k)
        frac = len(coords) / total
        spread = coords.std(0) / vol.shape[0] if len(coords) else np.zeros(3)
        feats.append((frac, spread))
    feats.sort(key=lambda f: -f[0])  # sort by volume -> modality-invariant ordering
    vec = []
    for frac, spread in feats:
        vec.append(frac)
        vec.extend(spread)
    return _l2(np.array(vec, np.float32))


EMBEDDERS = {
    "intensity": emb_intensity,
    "edges": emb_edges,
    "fingerprint": emb_fingerprint,
}


# --------------------------------------------------------------------------- scoring
def mrr(query_vecs, gallery_vecs, true_idx):
    """Mean reciprocal rank. query/gallery: (N,D) arrays; true_idx[i]=correct gallery row."""
    sims = query_vecs @ gallery_vecs.T          # cosine (vectors are L2-normalised)
    return mrr_from_scores(sims, true_idx)


def mrr_from_scores(scores, true_idx):
    """MRR from a precomputed (Nq,Ng) score matrix (higher = better). Used by pairwise rankers."""
    order = np.argsort(-scores, axis=1)
    rr = []
    for i, t in enumerate(true_idx):
        rank = int(np.where(order[i] == t)[0][0]) + 1
        rr.append(1.0 / rank)
    return float(np.mean(rr))


def evaluate():
    assert nib is not None, "nibabel is required for the real run (pip install nibabel)"
    rng = np.random.default_rng(SEED)
    index = build_image_index(DATA_ROOT)
    pairs = read_csv(DATA_ROOT / "dataset1" / "train_pairs.csv")
    rng.shuffle(pairs)
    val = pairs[:N_VAL]
    train = pairs[N_VAL:]
    print(f"DATA_ROOT={DATA_ROOT}  indexed={len(index)}  "
          f"val_pairs={len(val)}  train_pairs={len(train)}  grid={GRID}")

    # Optional trainable embedders: any module exposing build(pairs, index, grid, loader)
    # -> embed fn. Trained on the disjoint `train` split (no leakage), then scored like the
    # rest. Drop the module next to this file to enable it; absence is silently skipped.
    # Set SKIP_LEARNED=1 for a fast classical-only pass (the GPU rankers below need no training).
    plugins = []
    if os.environ.get("SKIP_LEARNED", "0") != "1":
        plugins.append(("learned_embedder", "learned"))
    if os.environ.get("USE_MIND_LEARNED", "1") == "1":      # MIND-input contrastive (d2/d3 model)
        plugins.append(("mind_embedder", "mind_learned"))
    for mod_name, key in plugins:
        try:
            mod = __import__(mod_name)
            EMBEDDERS[key] = mod.build(train, index, GRID, cached_volume)
        except Exception as e:  # noqa: BLE001 - never let a plugin break the references
            print(f"[skip] {key} embedder unavailable: {type(e).__name__}: {e}")

    # preload + cache normalised volumes for the held-out pairs
    q_raw, g_raw = [], []
    t0 = time.time()
    for i, p in enumerate(val):
        q_raw.append(cached_volume(p["query_id"], index[p["query_id"]], GRID))
        g_raw.append(cached_volume(p["target_id"], index[p["target_id"]], GRID))
        if (i + 1) % 10 == 0:
            print(f"  loaded {i + 1}/{len(val)} pairs  ({time.time() - t0:.0f}s)")

    # Pairwise GPU rankers (NMI / MIND / gradient cosine) — see rankers.py. These score a
    # query against a candidate jointly, so they can't be expressed as a single embedding;
    # they run on the MI300X GPU. Absence (no torch) is silently skipped.
    rankers = {}
    try:
        import rankers as rk
        rankers = rk.get_rankers()
        if rankers:
            print(f"[rankers] {list(rankers)} on device={rk.pick_device()}")
    except Exception as e:  # noqa: BLE001
        print(f"[skip] pairwise rankers unavailable: {type(e).__name__}: {e}")

    results = {}  # method -> {level -> mrr}
    true_idx = np.arange(len(val))
    for level, sim in SIMULATORS.items():
        # apply the level's simulator independently to each side (fixed rng per level)
        lvl_rng = np.random.default_rng(SEED + hash(level) % 1000)
        q_sim = [sim(v, lvl_rng) for v in q_raw]
        g_sim = [sim(v, lvl_rng) for v in g_raw]
        for name, emb in EMBEDDERS.items():
            qv = np.stack([emb(v) for v in q_sim])
            gv = np.stack([emb(v) for v in g_sim])
            score = mrr(qv, gv, true_idx)
            results.setdefault(name, {})[level] = score
            print(f"  [{level}] {name:12s} MRR={score:.3f}")
        for name, fn in rankers.items():
            scores = fn(q_sim, g_sim)
            score = mrr_from_scores(scores, true_idx)
            results.setdefault(name, {})[level] = score
            print(f"  [{level}] {name:12s} MRR={score:.3f}")

    write_results(results)
    return results


def write_results(results):
    levels = ["d1", "d2", "d3"]
    lines = ["# Offline validation results", "",
             f"- DATA_ROOT: `{DATA_ROOT}`",
             f"- held-out dataset1 pairs: **{N_VAL}**, grid {GRID}³, seed {SEED}",
             "- d1 = modality gap only · d2 = + deformation · d3 = + resection/bias/gamma",
             "- macro = mean(d1,d2,d3). Higher is better (random ≈ 1/gallery-ish).", "",
             "| embedder | d1 | d2 | d3 | **macro** |",
             "|---|---|---|---|---|"]
    for name, scores in sorted(results.items(), key=lambda kv: -np.mean(list(kv[1].values()))):
        macro = np.mean([scores[l] for l in levels])
        row = " | ".join(f"{scores[l]:.3f}" for l in levels)
        lines.append(f"| {name} | {row} | **{macro:.3f}** |")
    lines += ["", f"_generated {time.strftime('%Y-%m-%d %H:%M:%S')}_"]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate()
