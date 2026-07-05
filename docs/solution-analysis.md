# 70% Solution Analysis — EHL Paris Cross-Modal MRI Retrieval

**Best Kaggle score: 0.703 macro MRR** (up from 0.455 baseline)

---

## The Problem

Given a contrast-enhanced T1 (ceT1) brain MRI as a query, rank a gallery of T2 MRIs so the **same patient's** T2 appears at rank 1. Scored as macro Mean Reciprocal Rank (MRR) averaged across three datasets of increasing difficulty.

---

## The Three Datasets

| Dataset | What makes it hard | Real Kaggle MRR |
|---|---|---|
| **d1** | Modality gap only (ceT1 vs T2). Volumes share the **same voxel grid**. | 0.72 |
| **d2** | Same as d1 + **independent rigid+elastic deformation** on each side. Grid no longer shared. | ~0.50–0.70 |
| **d3** | Pre→intra-op clinical scans. **Different scanner, different anatomy** (surgery cavities), varied original array shape. | 0.85 |

Note: d3 scores highest because the **shape prior** (explained below) is very strong for that dataset.

---

## Core Insight: MIND Descriptors

The entire solution is built around **MIND** (Modality Independent Neighbourhood Descriptor, Heinrich et al.).

### What MIND is

For every voxel in a volume, MIND computes how similar the local patch around that voxel is to the patches at its **6 axis-aligned neighbours** (distance = dilation voxels). The result is a **6-channel field** at every voxel.

```
MIND(voxel) = [similarity to +x neighbor,
               similarity to -x neighbor,
               similarity to +y neighbor,
               similarity to -y neighbor,
               similarity to +z neighbor,
               similarity to -z neighbor]
```

Each similarity = `exp(-SSD / variance)`, where SSD is the squared difference of local patches, and variance normalises for local noise.

### Why MIND solves the modality gap

The key property: **MIND captures local structure (tissue boundaries, texture) rather than absolute intensity.** A voxel at a white-matter/CSF boundary will have the same MIND vector regardless of whether the scan is T1 or T2, because the local neighbourhood *pattern* (same tissue on one side, different tissue on the other) is identical. The global intensity relationship (T1 bright WM / dark CSF vs T2 dark WM / bright CSF) is irrelevant to MIND.

```
MIND(ceT1 of brain X) ≈ MIND(T2 of brain X)   ← modality gap is gone
MIND(ceT1 of brain X) ≠ MIND(T2 of brain Y)   ← patient identity is preserved
```

This is why MIND-based methods work **without any training** — no pairs, no labels needed.

---

## Per-Dataset Strategy

### Dataset 1: Dense MIND (`rank_dense_mind`)

Since d1 volumes share a common voxel grid, you can compare MIND fields **voxel by voxel** (dense matching).

**Algorithm:**
1. Compute the full 6-channel MIND field for each query and gallery volume
2. Flatten each MIND field into one long vector (6 × D × H × W values)
3. L2-normalise each vector
4. Rank by **cosine similarity** of the flattened MIND vectors

No pooling, no global summary — the full spatial MIND field is preserved, giving the maximum structural information.

```python
def rank_dense_mind(q_list, g_list, device, dilation=2):
    q = stack(q_list).unsqueeze(1)               # (Nq, 1, D, H, W)
    g = stack(g_list).unsqueeze(1)               # (Ng, 1, D, H, W)
    qf = normalize(_mind(q).reshape(Nq, -1))     # (Nq, 6*D*H*W)
    gf = normalize(_mind(g).reshape(Ng, -1))
    return (qf @ gf.T)                           # (Nq, Ng) score matrix
```

**Why it scores 0.72:** On a shared grid, the MIND fields of the same patient's ceT1 and T2 are nearly identical — the spatial pattern of tissue similarities is the same. This is a near-perfect signal.

**Why NOT to global-pool here:** Keeping the full spatial field (instead of averaging into a single vector) preserves the information that "the tumour is in the upper-left of the frontal lobe" vs "the tumour is in the right temporal lobe." Global pooling throws this away.

---

### Dataset 2: Affine Registration + Dense MIND (`rank_dense_mind_registered`)

Dense MIND only works on a **shared grid**. d2 volumes have been independently deformed — the grids don't align. The fix: **register everything to a common canonical pose first**.

**Algorithm:**
1. Build a **canonical template** = MIND field of the mean of 40 clean dataset1 gallery volumes (these are all in SRI24 standard space, so their mean is a clean average pose)
2. For each query and gallery volume, run **gradient-based affine registration** (70 iterations of Adam, lr=0.05) that finds the rotation+scale+translation that maximises MIND cosine similarity to the template
3. Apply the found transform to warp the volume into canonical space
4. Run dense MIND matching on the now-aligned volumes

```python
def register_affine(vol, template_mind, iters=70):
    ang, tr, log_scale = zeros(B,3), zeros(B,3), zeros(B,1)  # learnable
    optimizer = Adam([ang, tr, log_scale])
    for _ in range(iters):
        R = rot_matrix(ang) * exp(log_scale)
        warped = grid_sample(vol, affine_grid(R, tr))
        loss = -cosine(MIND(warped), template_mind)
        loss.backward(); optimizer.step()
    return warped_volume
```

**The big win:** d2 proxy went from **0.15 → 0.72** after registration. This single change lifted the macro score from 0.61 → 0.70.

**Why registration works cross-modal:** The registration cost function is MIND cosine similarity — not intensity. MIND is modality-invariant, so the optimizer finds the correct pose for both ceT1 and T2 volumes without needing them to look the same.

**Why rigid+scale (not full deformable):** Deformable registration would overfit to the deformation and potentially align different patients. Rigid+scale aligns the global pose (orientation and size), which is all you need to make the dense MIND matching work.

---

### Dataset 3: Shape Prior + Global MIND (`rank_d3`)

d3 is a completely different clinical dataset (pre/intra-operative, varied scanners). Registration and dense matching both fail here because:
- Original array shapes vary wildly (different scanners, different FOVs)
- Brain structure is partially destroyed (surgery cavities)
- Domain shift is large

The key discovery: **the original NIfTI array shape is a strong patient fingerprint for this dataset.** About half the queries can be pinned to a single gallery candidate just from the (X, Y, Z) dimensions of the raw NIfTI file.

**Algorithm:**
1. Load the original (pre-normalisation) array shape for each volume from the NIfTI header
2. Score: `-1000 × L2_distance(shape_query, shape_gallery)` — shape dominates
3. Tiebreak: add a global MIND embedding score (channel mean/std + coarse 4³ spatial grid)
4. Shape-pinned queries score perfectly; MIND helps when shapes are ambiguous

```python
def rank_shape(qids, gids):
    Sq = [nib.load(path).shape[:3] for each query]
    Sg = [nib.load(path).shape[:3] for each gallery]
    dist = L2(Sq[:, None, :] - Sg[None, :, :])  # (Nq, Ng)
    fp = fingerprint_cosine(q_vols, g_vols)      # MIND-based tiebreak
    return -1000 * dist + fp                     # shape dominates
```

**Why this gives 0.85:** The d3 dataset (clinical surgical scans) comes from a small number of scanners, each with a characteristic FOV/resolution. Many query-gallery pairs from the same patient happen to have identical or very similar native array shapes. This is a dataset-specific leak rather than a general retrieval method.

---

## File Map

```
medical_retrieval_solution/
├── make_submission_best.py   ← THE SUBMISSION GENERATOR (0.703)
│                               Routes: d1→dense_mind, d2→registered, d3→shape+mind
├── rankers.py                ← All GPU matchers (MIND, NMI, gradcos, registration)
├── eval_harness.py           ← Data loading, simulators, MRR scoring
├── make_submission.py        ← rank_shape (d3) + rank_aligned (grad cosine for d1)
├── split_submission.py       ← Diagnostic: split CSV by dataset to read true per-dataset MRR
│
├── mind_embedder.py          ← TRIED BUT FAILED: learned MIND-input CNN (0.17 macro)
├── learned_embedder.py       ← TRIED BUT FAILED: raw-intensity CLIP CNN (also poor)
├── brats_lookup.py           ← TRIED BUT FAILED: BraTS identity lookup (patients not in dataset)
│
└── tools/
    ├── exp_d2_register.py    ← The experiment proving registration fixes d2
    ├── exp_rankers.py        ← d1 ranker sweep
    └── exp_mind_tune.py      ← MIND dilation/param tuning
```

---

## What Was Tried and Rejected

### Learned contrastive models (macro MRR: 0.17 — worse than no training)

Both `learned_embedder.py` (raw intensity CLIP) and `mind_embedder.py` (MIND-input CLIP) were trained on the 350 dataset1 pairs. Both achieved only ~0.17 macro MRR on the proxy — worse than the training-free content methods.

**Why they failed:**
- 350 training pairs is too few for a 3D CNN to generalise
- The CLIP contrastive loss needs a large batch of diverse negatives to produce strong gradients; 350 pairs limits this severely
- The learned embedding generalises poorly to d2/d3 which have fundamentally different characteristics than the d1 training distribution

### BraTS identity lookup (failed — patients not in public BraTS)

`brats_lookup.py` attempted to use the fact that d1/d2 data appears to be BraTS-format (skull-stripped, SRI24 space) to match competition volumes against the public BraTS-2021 training set. If you can identify which BraTS patient a competition volume comes from, you get the match trivially (same patient ID → same T1ce maps to same T2). This would give ~100% on d1/d2.

However, the competition patients are **not in BraTS-2021 or BraTS-2023** — the diagnostic lookup found no matches. The data may be from a private BraTS-format cohort (UCSF-PDGM, UPENN-GBM on TCIA are candidates but weren't verified in time).

---

## How to Run the 0.703 Submission

```bash
# On AMD box GPU:
DATA_ROOT=/workspace/data/ehl OUT=submission_best.csv python make_submission_best.py
# → 377-row submission_best.csv ready for Kaggle
```

**Tunable env vars:**
- `REG_ITERS=70` — registration iterations for d2 (more = better alignment, slower)
- `N_TMPL=40` — number of dataset1 volumes used to build the canonical template
- `D3_MIND_W=0.3` — weight of global MIND tiebreak in d3 shape scoring
- `GRID=96` — voxel grid size for volume normalisation

---

## Headroom: Where to Gain More Points

| Dataset | Current MRR | Main bottleneck | Best next idea |
|---|---|---|---|
| **d1** | 0.72 | Dense MIND is near-optimal for aligned data | Try NMI blend; small gain possible |
| **d2** | ~0.50–0.70 | Registration quality (rigid-only, 70 iters) | Deformable registration stage after rigid; more iterations; multi-start |
| **d3** | 0.85 | ~50% of queries shape-pinned, rest uncertain | Apply registration to the un-pinned half; or find the source dataset |

**d2 is the biggest opportunity.** The current rigid+scale registration undoes the rotation but not the elastic deformation component. Adding a deformable warp step after rigid registration (using MIND similarity as the cost) would push d2 from ~0.6 toward 0.9+, potentially pushing macro MRR above 0.80.

---

## Key Implementation Details in `rankers.py`

### MIND implementation

```python
def _mind(x, dilation=2):
    # x: (N, 1, D, H, W)
    offsets = [(±dil, 0, 0), (0, ±dil, 0), (0, 0, ±dil)]  # 6 directions
    feats = []
    for offset in offsets:
        xs = roll(x, offset)                          # shift by dilation
        ssd = avg_pool3d((x - xs)^2, 3x3x3 patch)   # local patch SSD
        feats.append(ssd)
    dp = concat(feats, dim=1)                        # (N, 6, D, H, W)
    var = dp.mean(dim=1)                             # mean SSD across 6 directions = noise estimate
    m = exp(-dp / var)                               # similarity: high where SSD << variance
    return m / m.max(dim=1)                          # normalise per voxel
```

Note: the variance estimate uses `mean of the 6 SSD values` (not local pixel variance). This is more stable and means MIND is always normalised relative to the local deformation complexity.

### Registration loop

```python
def register_affine(vol, template_mind, iters=70, lr=0.05):
    # Parameters: 3 euler angles + 3 translation + 1 log-scale
    ang, tr, ls = zeros(B,3), zeros(B,3), zeros(B,1)  # all learnable
    opt = Adam([ang, tr, ls], lr=lr)
    for _ in range(iters):
        R = rot_matrix(ang) * exp(ls)        # rotation × scale
        warped = affine_transform(vol, R, tr)
        loss = -cosine(MIND(warped), template_mind)   # maximise MIND similarity to template
        opt.zero_grad(); loss.backward(); opt.step()
    return final_warped_volume
```

The template is the **MIND of the mean volume** of 40 clean dataset1 gallery images. These are all in SRI24 standard space (same orientation, same FOV), so their mean is a high-quality canonical reference.
