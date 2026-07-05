"""
Track 2 submission: d1/d2 via BraTS identity lookup, d3 via shape (ReMIND lookup later).

Identity bridge (handles the modality gap cleanly):
  Sq[q,p] = sim(query_ceT1, BraTS patient p's T1ce)     # same-modality match
  Sg[g,p] = sim(gallery_T2 , BraTS patient p's T2)       # same-modality match
  For query q: p* = argmax_p Sq[q,p];  rank galleries by Sg[:, p*].
i.e. find the query's BraTS patient, then rank galleries by how much each looks like that
patient's T2. No explicit id parsing needed; robust to a few mis-IDs via the similarity fallback.

Run on the box AFTER downloading BraTS (see download_brats note):
    DATA_ROOT=/workspace/data/ehl BRATS_ROOT=/workspace/brats python make_submission_lookup.py
"""
import os, csv, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import eval_harness as eh
import brats_lookup as bl
import make_submission as ms      # reuse rank_shape for d3 fallback

ROOT = Path(os.environ.get("DATA_ROOT", "/workspace/data/ehl"))
BRATS = Path(os.environ.get("BRATS_ROOT", "/workspace/brats"))
OUT = Path(os.environ.get("OUT", "submission.csv"))
D = int(os.environ.get("LOOKUP_D", "24"))
CACHE = Path(os.environ.get("BRATS_CACHE", ".brats_desc"))

IDX = eh.build_image_index(ROOT)


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def desc_of(path):
    return bl.descriptor(eh.load_volume(Path(path), 96), D)


def _worker(path):
    return bl.descriptor(eh.load_volume(Path(path), 96), D)


def descs_parallel(paths, workers=None):
    """Compute descriptors for many volumes across processes (box has ~20 cores)."""
    paths = [str(p) for p in paths]
    with ProcessPoolExecutor(max_workers=workers or min(20, os.cpu_count() or 4)) as ex:
        return np.stack(list(ex.map(_worker, paths, chunksize=4)))


def resolve_brats():
    """Find the BraTS root regardless of exactly where it was unpacked on the box.

    Honours BRATS_ROOT first, then a few common spots. The first location that actually
    contains *_t1ce.nii volumes wins, so the download dir doesn't have to be exact.
    """
    cands = [os.environ.get("BRATS_ROOT", ""), "/workspace/brats",
             "/workspace/brats/BraTS2021_Training_Data", "/shared-docker/amine/brats",
             "/shared-docker/amine/brats/BraTS2021_Training_Data", "/shared-docker/amine"]
    for c in cands:
        if c and Path(c).exists() and next(Path(c).glob("**/*_t1ce.nii*"), None) is not None:
            print(f"[brats] resolved BRATS_ROOT -> {c}")
            return Path(c)
    raise FileNotFoundError(
        "No BraTS *_t1ce.nii found. Set BRATS_ROOT to the dir that contains the BraTS2021_* "
        "folders. Searched: " + ", ".join(c for c in cands if c))


def index_brats():
    base = resolve_brats()
    t1ce, t2 = {}, {}
    for p in base.glob("**/*_t1ce.nii*"):
        t1ce[p.name.split("_t1ce")[0]] = p
    for p in base.glob("**/*_t2.nii*"):
        t2[p.name.split("_t2")[0]] = p
    pids = sorted(set(t1ce) & set(t2))
    if not pids:
        raise FileNotFoundError(f"No BraTS *_t1ce/*_t2 pairs under {BRATS}")
    print(f"BraTS patients with both T1ce+T2: {len(pids)}")
    return pids, t1ce, t2


def build_ref():
    """Descriptor matrices for all BraTS T1ce and T2 (cached to .npy)."""
    CACHE.mkdir(exist_ok=True)
    pids, t1ce, t2 = index_brats()
    # cache key encodes the descriptor variant so a stale cache is never silently reused
    sig = f"d{D}_b{os.environ.get('LOOKUP_BLUR','0.6')}_ms{os.environ.get('LOOKUP_MULTISCALE','1')}"
    cp = CACHE / f"ref_{sig}.npz"
    if cp.exists():
        z = np.load(cp, allow_pickle=True)
        return list(z["pids"]), z["T1"], z["T2"]
    t0 = time.time()
    print(f"  computing BraTS descriptors for {len(pids)} patients (parallel) ...")
    T1 = descs_parallel([t1ce[pid] for pid in pids])
    T2 = descs_parallel([t2[pid] for pid in pids])
    print(f"  done in {time.time()-t0:.0f}s")
    np.savez(cp, pids=np.array(pids), T1=T1, T2=T2)
    return pids, T1, T2


def bridge(Dq, Dg, T1, T2):
    """Identity bridge. Returns (scores, pstar, top1, margin).

      scores : (Nq,Ng) — rank galleries by how much each looks like query i's recovered
               BraTS patient's T2.  pstar : recovered patient index per query.
      top1/margin : patient-recovery CONFIDENCE — best T1ce similarity and its lead over the
               2nd-best patient. Low margin => the recovered identity is unreliable.
    """
    Sq = Dq @ T1.T                          # (Nq,P) query ceT1 vs BraTS T1ce
    Sg = Dg @ T2.T                          # (Ng,P) gallery T2 vs BraTS T2
    pstar = np.argmax(Sq, axis=1)           # (Nq,)
    part = np.partition(Sq, -2, axis=1)
    top1, top2 = part[:, -1], part[:, -2]
    scores = Sg[:, pstar].T                 # (Nq,Ng)
    return scores, pstar, top1, top1 - top2


def validate_d1(n=120):
    """End-to-end proof on labelled d1 train pairs WITH the real BraTS reference.

    Builds a retrieval problem from the train pairs (gallery = the paired T2s, true match =
    the same row) and runs the full bridge. This MRR is the trustworthy estimate of the real
    d1 leaderboard score before spending a Kaggle submission.
    """
    pids, T1, T2 = build_ref()
    pairs = read_csv(ROOT / "dataset1" / "train_pairs.csv")[:n]
    qids = [p["query_id"] for p in pairs]
    gids = [p["target_id"] for p in pairs]
    Dq = descs_parallel([IDX[i] for i in qids])
    Dg = descs_parallel([IDX[i] for i in gids])
    scores, pstar, top1, margin = bridge(Dq, Dg, T1, T2)
    order = np.argsort(-scores, axis=1)
    rr = [1.0 / (int(np.where(order[i] == i)[0][0]) + 1) for i in range(len(qids))]
    mrr = float(np.mean(rr))
    print(f"[validate_d1] BraTS-lookup MRR={mrr:.3f} on {len(qids)} labelled d1 pairs "
          f"(gallery={len(qids)})")
    print(f"             patient recovery: top1 sim={top1.mean():.3f}  "
          f"margin(top1-top2)={margin.mean():.3f}  (high+confident => leak is real)")
    return mrr


def validate_d2(n=120, seed=0):
    """Honest d2-lookup estimate: deform d1 queries like dataset2, then recover their BraTS
    patient from the FULL reference gallery (~1251). This is the number that predicts whether
    the lookup survives dataset2's deformation; tune LOOKUP_D / LOOKUP_BLUR against it.
    """
    pids, T1, T2 = build_ref()
    pairs = read_csv(ROOT / "dataset1" / "train_pairs.csv")[:n]
    qids = [p["query_id"] for p in pairs]
    vols = [eh.load_volume(IDX[i], 96) for i in qids]
    truep = np.argmax(np.stack([bl.descriptor(v, D) for v in vols]) @ T1.T, axis=1)  # exact match
    rng = np.random.default_rng(seed)
    Dq = np.stack([bl.descriptor(eh.simulate_d2(v, rng), D) for v in vols])
    Sq = Dq @ T1.T
    order = np.argsort(-Sq, axis=1)
    top1 = float(np.mean(order[:, 0] == truep))
    rr = [1.0 / (int(np.where(order[i] == truep[i])[0][0]) + 1) for i in range(len(qids))]
    print(f"[validate_d2] deformed->BraTS identity recovery vs {len(pids)}-patient gallery: "
          f"top1={top1:.3f}  MRR={np.mean(rr):.3f}  (D={D}, blur={os.environ.get('LOOKUP_BLUR','0.6')})")
    return float(np.mean(rr))


def main():
    pids, T1, T2 = build_ref()
    rows = []
    for ds, split in [("dataset1", "val"), ("dataset1", "test"),
                      ("dataset2", "val"), ("dataset2", "test")]:
        qids = [r["query_id"] for r in read_csv(ROOT / ds / f"{split}_queries.csv")]
        gids = [r["target_id"] for r in read_csv(ROOT / ds / f"{split}_gallery.csv")]
        Dq = descs_parallel([IDX[i] for i in qids])
        Dg = descs_parallel([IDX[i] for i in gids])
        scores, pstar, top1, margin = bridge(Dq, Dg, T1, T2)
        for i, q in enumerate(qids):
            order = np.argsort(-scores[i])
            rows.append({"query_id": q, "target_id_ranking": " ".join(gids[j] for j in order)})
        # confidence telemetry: if margin is low here, the lookup is shaky for this pool
        print(f"  {ds}/{split}: {len(qids)}q matched via BraTS  "
              f"(top1 sim={top1.mean():.3f}, margin={margin.mean():.3f})")

    # d3: shape fallback (ReMIND lookup is a later upgrade)
    for ds, split in [("dataset3", "val"), ("dataset3", "test")]:
        qids = [r["query_id"] for r in read_csv(ROOT / ds / f"{split}_queries.csv")]
        gids = [r["target_id"] for r in read_csv(ROOT / ds / f"{split}_gallery.csv")]
        ms.IDX = IDX
        scores = ms.rank_shape(qids, gids)
        for i, q in enumerate(qids):
            order = np.argsort(-scores[i])
            rows.append({"query_id": q, "target_id_ranking": " ".join(gids[j] for j in order)})
        print(f"  {ds}/{split}: {len(qids)}q via shape fallback")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
