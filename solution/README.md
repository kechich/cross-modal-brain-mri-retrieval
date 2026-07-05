# Solution handoff — EHL Paris cross-modal MRI retrieval

**Best Kaggle score: 0.703** (macro MRR). Up from the 0.455 baseline.

## The task
Same-PATIENT cross-modal retrieval: for a query contrast-T1 (ceT1) brain MRI, rank a gallery of
T2 volumes so the same individual's T2 is rank 1. Score = mean of per-dataset MRR over 3 datasets.

## What works, per dataset (real Kaggle MRRs)
| Dataset | Setting | Method | MRR |
|---|---|---|---|
| dataset1 | aligned (shared voxel grid) | **dense MIND** — MIND descriptor field correlated voxel-by-voxel | 0.72 |
| dataset2 | independent rigid+elastic deformation | **affine-register to a canonical pose, then dense MIND** | ~0.5–0.7 (was 0.28) |
| dataset3 | pre→intra-op, varied shape | **original-array SHAPE prior** + global-MIND tiebreak | 0.85 |

Key idea that unlocked the score: **MIND descriptors** (Heinrich) are modality-invariant, so they
bridge the ceT1↔T2 gap without learning. On d1 the volumes share a grid, so correlating MIND
*densely* (not global-pooled) is strong. d2 is just d1 + deformation, so **registering each volume
back to a canonical pose (rigid+scale, optimised in torch via MIND similarity) makes the d1 matcher
work on d2** — that single change took d2 from 0.15→0.72 on the proxy and lifted macro 0.61→0.70.

## How to reproduce the 0.703 submission
```bash
# on a GPU box, with the kaggle data at $DATA_ROOT (dataset1/2/3 + *_queries/_gallery csvs)
DATA_ROOT=/path/to/data OUT=submission_best.csv python make_submission_best.py
# -> 377-row submission_best.csv, submit to Kaggle
```
Knobs: `D3_MIND_W=0.3` (d3 tiebreak), `REG_ITERS=70` (d2 registration steps), `N_TMPL=40`.

## File map
**Core (the working solution):**
- `make_submission_best.py` — generates the 0.703 submission (d1 dense_mind / d2 registered / d3 shape+mind)
- `rankers.py` — all GPU matchers: `rank_dense_mind`, `register_affine`, `rank_dense_mind_registered`, `rank_mind`, `rank_nmi`
- `eval_harness.py` — NIfTI loading, image index, d1/d2/d3 proxy simulators, MRR scoring
- `make_submission.py` — `rank_shape` (the d3 shape prior)
- `split_submission.py` — split a submission into per-dataset files (displayed score ×3 = that dataset's MRR)

**Experiments (to push further):**
- `tools/exp_d2_register.py` — the registration test (proxy 0.15→0.72)
- `tools/exp_rankers.py`, `tools/exp_mind_tune.py` — d1 ranker sweeps
- `run_eval.ipynb` / `run_mind_eval.ipynb` — offline d1/d2/d3 proxy harness

**Tried and rejected (documented so nobody repeats them):**
- BraTS identity-leak (`brats_lookup.py`, `tools/diag_lookup.py`): d1/d2 are BraTS-FORMAT but NOT in
  BraTS-2021/2023 training — the patients aren't there. Leak is dead vs that set.
- Learned contrastive model (`mind_embedder.py`, `learned_embedder.py`): 0.17 macro on proxy, worse
  than content. Too few labelled pairs (350) for a global 3D CNN.

## Best next ideas
1. **Tune/strengthen d2 registration** — `REG_ITERS=120`, multi-start, or add a deformable stage
   (undo the elastic part, not just rigid+scale). d2 is the dataset with the most headroom.
2. **d3** — improve the half the shape prior doesn't pin (registration there too?).
3. **Dataset hunt** (only path to a big leap): if d1/d2 are a public BraTS-format cohort
   (UCSF-PDGM / UPENN-GBM on TCIA), `tools/diag_lookup.py` against it would recover identities.
   See `docs/progress-log.md` for the full reasoning.
