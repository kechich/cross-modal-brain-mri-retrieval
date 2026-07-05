# Offline validation results

- DATA_ROOT: `/workspace/data/ehl`
- held-out dataset1 pairs: **60**, grid 96³, seed 20260627
- d1 = modality gap only · d2 = + deformation · d3 = + resection/bias/gamma
- macro = mean(d1,d2,d3). Higher is better (random ≈ 1/gallery-ish).

| embedder | d1 | d2 | d3 | **macro** |
|---|---|---|---|---|
| learned | 0.833 | 0.501 | 0.338 | **0.557** |
| intensity | 0.350 | 0.150 | 0.089 | **0.196** |
| fingerprint | 0.186 | 0.142 | 0.125 | **0.151** |
| edges | 0.087 | 0.087 | 0.076 | **0.083** |

_generated 2026-06-27 17:49:27_
