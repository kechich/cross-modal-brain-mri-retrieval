# Your Action Plan — Mohamed Folder Setup

## What's Ready (on AMD box in `/shared-docker/mohamed`)

1. **ANALYSIS.md** — Full breakdown of:
   - Current architecture (CLIP-style 3D CNN)
   - Current performance (0.833 / 0.501 / 0.338 on d1/d2/d3 → macro 0.557)
   - Gap analysis: why d2 drops 40%, d3 drops 67%
   - Three improvement tiers (quick wins → advanced)

2. **learned_embedder_v2.py** — Tier 1 improvements already built:
   - **Network**: width 16 → 24 (+3.6× capacity)
   - **Augmentation**: rot 15° → 25°, elastic 0.06 → 0.10, added flips, stronger translation
   - **Loss**: CLIP → triplet loss with hard negative mining
   - **Expected**: 0.557 → 0.59+ macro

3. **test_embedder_v2.ipynb** — Ready to run:
   - Train v1 and v2 on the same 290 pairs
   - Compare d1/d2/d3 scores
   - Tells you if the improvements are working

## What You Need to Do NOW

### Step 1: Run the test (20 min)
On the AMD box in JupyterLab:
```
1. Open /shared-docker/mohamed/test_embedder_v2.ipynb
2. Run All (cells execute sequentially)
3. Watch progress — training takes ~5 min per model
```

**Expected output:**
- V1 d1 ≈ 0.833 (baseline check)
- V2 d1 ≈ 0.85+ (should be slightly better)
- If V2 > V1: **proceed to step 2**
- If V2 ≤ V1: something's broken; we'll debug

### Step 2: Full eval if step 1 succeeds (1 hour)
Once v2 shows improvement on d1, run the full 3-level eval:
```bash
# On AMD box terminal in /shared-docker/amine:
DATA_ROOT=/workspace/data/ehl python eval_harness.py --embedder-module /shared-docker/ahmed/learned_embedder_v2
# OR: modify run_eval.ipynb to use v2 instead of v1
```

Expected: v2 macro ≈ 0.59–0.61 (vs v1 = 0.557, 6–10% gain)

### Step 3: If v2 works, decide on next move
- **If 0.59–0.61 macro achieved**: 
  → Proceed to Tier 2 (test-time augmentation + refined loss)
  → Submit v2 to Kaggle for public LB score
  
- **If < 0.59 macro**: 
  → Revert; try Tier 2 directly (TTA helps d2/d3 more)
  → Or try ensemble of v1+v2

## Where Improvements Will Come From

| Lever | Expected gain | Effort | Next? |
|---|---|---|---|
| **v2 alone** (current) | +3–5% | Done | ✓ Test now |
| **TTA** (avg 8 augmented evals) | +2–3% (esp d2) | 20 min | If v2 works |
| **Loss tuning** (focal, margin) | +1–2% | 30 min | If TTA helps |
| **Coarse alignment** (for d3) | +1–2% (d3 only) | 1 hour | Endgame |
| **Ensemble** (3–5 seeds) | +2–3% | 2 hours | Final push |

## Files & Folders
```
/shared-docker/
├── amine/              ← colleagues' work (baseline)
│   ├── eval_harness.py
│   ├── learned_embedder.py
│   ├── run_eval.ipynb
│   └── eval_results.md
├── mohamed/            ← YOUR WORK
│   ├── ANALYSIS.md
│   ├── learned_embedder_v2.py
│   ├── test_embedder_v2.ipynb
│   └── (results go here)
└── (data symlinked to /workspace/data/ehl)
```

## Git Workflow
Everything is already committed locally. After each successful improvement:
```bash
git add -A
git commit -m "Describe improvement + expected gain"
git push origin main
```

The `entire/checkpoints/v1` branch auto-captures everything for grading.

## Deadline Reminder
**June 28, 10:30 UTC** — ~24 hours away.
- Use time for testing (v2) + one more improvement (TTA or ensemble)
- No time for architecture rewrite; focus on tuning current setup

Good luck! Start with step 1 now. 🚀
