# ML Architecture Analysis & Improvement Strategy

## Current State

**Colleagues have built:**
- `learned_embedder.py`: CLIP-style 3D CNN with deformation-invariant training
- `eval_harness.py`: Synthetic d1/d2/d3 proxy evaluation (modality gap → deformation → structural change)
- Baseline: tiny 2D slice-CLIP (reference benchmark)

**Current Performance (Offline d1/d2/d3 proxies):**
```
d1 (modality gap only):       0.833 MRR
d2 (+ deformation):           0.501 MRR  ← 40% drop
d3 (+ structural change):     0.338 MRR  ← 67% drop from d1
Macro (mean):                 0.557
```

## Architecture Details

### Encoder: `Encoder3D` (compact 3D CNN)
```
Input (96³)
  → Conv3d(1→16, stride=2)   [48³]
  → Conv3d(16→32, stride=2)  [24³]
  → Conv3d(32→64, stride=2)  [12³]
  → Conv3d(64→128, stride=2) [6³]
  → GlobalAvgPool → Linear(128→128)
  → L2-normalized embedding
```

**Issue**: Very compact (total ~270k params). May underfit on 290 training pairs.

### Training: CLIP-style symmetric loss
```python
logits = scale * z_query @ z_target.T
loss = (CE(logits, labels) + CE(logits.T, labels)) / 2
```
- Only optimizes pairwise positive-negative contrast (no hard negatives)
- Symmetric loss helps with small batch sizes but limited signal

### Augmentation (INDEPENDENT per query/target)
**Geometric:**
- Rotation: ±15° (random axis)
- Scale: ±10%
- Translation: ±12% of volume size
- Elastic warp: 5×5×5 random control grid, σ=0.06

**Intensity:**
- Gamma: 0.6–1.6
- Contrast inversion: 50% flip (ceT1↔T2 cue)
- Bias field: smooth multiplicative (0.7–1.3)
- Noise: σ=0.03

## Performance Gaps

### Gap 1: d2 (Deformation) — 40% loss
**Why**: Current augmentation helps but model still fragile to untrained deformations.

**Signals**:
- Geometric aug params (rot=15°, elastic=0.06) may be too weak
- Network doesn't learn truly deformation-invariant features
- No spatial normalization / alignment preprocessing

### Gap 2: d3 (Structural Change) — 67% loss from d1
**Why**: Resection cavities (missing anatomy), domain shift (different hospital), pre-post surgery intensity patterns are fundamentally different.

**Signals**:
- Pure learned embeddings can't bridge structural change without guidance
- Need coarse alignment or global anatomical priors
- Intensity aug helps (0.196→0.338 over baseline) but not enough

## Improvement Strategy

### Tier 1: Quick wins (implement in 1 hour)
1. **Stronger geometric augmentation**
   - Increase `EMB_ROT` from 15° → 25°
   - Increase `EMB_ELASTIC` from 0.06 → 0.10
   - Add random flip (50% x/y/z flip)
   - Test: rerun eval_harness, expect 2–5% d2 improvement

2. **Network capacity**
   - Increase base width: `EMB_WIDTH` from 16 → 24 (params: 270k → 760k)
   - Add residual connections in encoder blocks
   - Test: should help d1 directly

3. **Loss function swap**
   - Implement triplet loss with hard negative mining
   - Or try ArcFace-style margin
   - Should improve all three levels

### Tier 2: Medium effort (2–3 hours)
1. **Multi-scale embeddings**
   - Extract features from intermediate layers (32, 64, 128 channels)
   - Concatenate before final projection
   - Expected: richer feature representation, esp. for d3

2. **Test-time augmentation (TTA)**
   - Encode each query/target 8 times with different augmentations
   - Average embeddings
   - Should help d2 robustness significantly

3. **Coarse alignment preprocessing (for d3)**
   - Center of mass alignment before encoding
   - Or learnable registration (if time permits)

### Tier 3: Advanced (if time allows)
1. **Separate encoders with alignment loss**
   - One for ceT1, one for T2
   - Align their embeddings via contrastive loss
   - May outperform single shared encoder on modality-specific features

2. **Intensity-invariant features**
   - Preprocess with histogram equalization / standardization
   - Learn on Sobel/Laplacian edges instead of raw intensity
   - Helps with d3 domain shift

3. **Ensemble multiple seeds**
   - Train 3–5 models with different random seeds
   - Blend embeddings at inference
   - Proven to add 2–3% MRR

## Recommended Start

**Phase 1 (Now): Quick augmentation + capacity test**
- Bump `EMB_ROT`, `EMB_ELASTIC`, `EMB_WIDTH`
- Run `eval_harness.py` offline (no Kaggle submissions)
- Should see if we get 0.55 → 0.59 macro

**Phase 2 (If phase 1 works): TTA + triplet loss**
- Implement TTA
- Swap loss function
- Should push toward 0.61–0.63

**Phase 3 (Endgame): Only if ahead of time → Coarse alignment for d3**
- Register volumes before encoding
- Can add 1–2% on d3 alone

## Files to Modify
- `learned_embedder.py`: Config, model, loss, TTA
- `eval_harness.py`: Run evals locally (no GPU needed for test)
- `run_eval.ipynb`: Visualization + metrics (on AMD box)

## Success Metrics
- **Target macro**: 0.60+ (from 0.557)
- **Breakdown**: d1>0.85, d2>0.55, d3>0.39
