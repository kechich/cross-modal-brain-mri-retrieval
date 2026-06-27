"""Local smoke test for eval_harness logic — no real data / nibabel needed.

Fabricates N distinct synthetic 'brain' volumes, then checks:
  * simulators run and preserve shape / range
  * embedders return finite L2-normalised vectors
  * MRR == 1.0 when query==gallery (identity sanity)
  * on d1-proxy each embedder ranks the matching synthetic volume well
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import eval_harness as eh

GRID = eh.GRID
rng = np.random.default_rng(0)


def fake_volume(seed):
    """A blobby synthetic volume with a unique per-identity structure."""
    r = np.random.default_rng(seed)
    v = np.zeros((GRID, GRID, GRID), np.float32)
    zz, yy, xx = np.ogrid[:GRID, :GRID, :GRID]
    for _ in range(6):
        c = r.uniform(GRID * 0.3, GRID * 0.7, 3)
        rad = r.uniform(GRID * 0.08, GRID * 0.2)
        blob = ((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) < rad ** 2
        v[blob] += r.uniform(0.3, 1.0)
    return np.clip(v / (v.max() + 1e-6), 0, 1).astype(np.float32)


def main():
    N = 12
    vols = [fake_volume(i) for i in range(N)]

    # simulator sanity
    for name, sim in eh.SIMULATORS.items():
        out = sim(vols[0], np.random.default_rng(1))
        assert out.shape == vols[0].shape, f"{name} changed shape"
        assert np.isfinite(out).all(), f"{name} produced non-finite"
        assert 0.0 <= out.min() and out.max() <= 1.0 + 1e-5, f"{name} out of range"
    print("simulators: OK")

    # embedder sanity + identity MRR
    for name, emb in eh.EMBEDDERS.items():
        vecs = np.stack([emb(v) for v in vols])
        assert np.isfinite(vecs).all(), f"{name} non-finite vec"
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms[norms > 0], 1.0, atol=1e-4), f"{name} not L2-normalised"
        score = eh.mrr(vecs, vecs, np.arange(N))
        assert score == 1.0, f"{name} identity MRR={score} (expected 1.0)"
        print(f"  {name:12s} dim={vecs.shape[1]:3d} identity-MRR=1.000")

    # d1 proxy: query==gallery==same identities -> should be perfect/near-perfect
    print("d1-proxy MRR (same vols, no perturbation):")
    for name, emb in eh.EMBEDDERS.items():
        qv = np.stack([emb(v) for v in vols])
        score = eh.mrr(qv, qv, np.arange(N))
        print(f"  {name:12s} {score:.3f}")

    # d2 proxy on synthetic data: deformation should degrade intensity more than fingerprint
    print("d2-proxy MRR (independent deformation on each side):")
    for name, emb in eh.EMBEDDERS.items():
        r = np.random.default_rng(7)
        qv = np.stack([emb(eh.simulate_d2(v, r)) for v in vols])
        gv = np.stack([emb(eh.simulate_d2(v, r)) for v in vols])
        score = eh.mrr(qv, gv, np.arange(N))
        print(f"  {name:12s} {score:.3f}")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
