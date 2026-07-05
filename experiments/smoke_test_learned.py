"""Local CPU smoke test for learned_embedder — tiny grid, 2 epochs, synthetic data.

Checks: augmentation preserves shape, training loop runs, embed() returns a finite
L2-normalised vector of the right dim. No real data / GPU needed.
"""
import os, sys
os.environ.update(EMB_EPOCHS="2", EMB_BATCH="4", EMB_DIM="32", EMB_WIDTH="8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import learned_embedder as le

GRID = 24


def fake_vol(seed):
    r = np.random.default_rng(seed)
    v = np.zeros((GRID, GRID, GRID), np.float32)
    zz, yy, xx = np.ogrid[:GRID, :GRID, :GRID]
    for _ in range(4):
        c = r.uniform(GRID * 0.3, GRID * 0.7, 3)
        v[((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) < (GRID * 0.15) ** 2] += 1
    return np.clip(v / (v.max() + 1e-6), 0, 1).astype(np.float32)


def main():
    import torch
    # augmentation shape check
    x = torch.from_numpy(np.stack([fake_vol(i) for i in range(4)])[:, None]).float()
    a = le.augment(x, le._cfg())
    assert a.shape == x.shape, f"augment changed shape {a.shape} vs {x.shape}"
    assert torch.isfinite(a).all(), "augment produced non-finite"
    assert float(a.min()) >= 0 and float(a.max()) <= 1.0 + 1e-4, "augment out of [0,1]"
    print("augment: OK", tuple(a.shape))

    # fake dataset: 10 pairs, loader returns a fake volume per id
    pairs = [{"query_id": f"q{i}", "target_id": f"g{i}"} for i in range(10)]
    index = {p[k]: p[k] for p in pairs for k in ("query_id", "target_id")}
    store = {iid: fake_vol(hash(iid) % 9999) for iid in index}
    loader = lambda iid, path, grid: store[iid]

    embed = le.build(pairs, index, GRID, loader, device="cpu")
    v = embed(fake_vol(123))
    assert v.shape == (32,), f"embed dim {v.shape}"
    assert np.isfinite(v).all(), "embed non-finite"
    assert abs(np.linalg.norm(v) - 1.0) < 1e-4, f"not L2-normalised: {np.linalg.norm(v)}"
    print("build + embed: OK  dim=", v.shape[0])
    print("\nLEARNED SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
