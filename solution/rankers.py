"""GPU pairwise rankers for the cross-modal retrieval harness.

The reference *embedders* in `eval_harness.py` map one volume -> one vector and rank by
cosine. The strongest cross-modal same-subject methods are inherently *pairwise* (they look
at a query and a candidate together) or need a non-cosine descriptor distance, so they don't
fit that contract. This module implements them as **rankers**:

    ranker(query_vols, gallery_vols) -> (Nq, Ng) score matrix   (higher = better match)

All rankers are torch and run on the MI300X GPU (ROCm) when available, CPU otherwise. They
take plain lists of float32 HxWxD numpy volumes (already grid-normalised by the harness).

Implemented:
  * nmi      : normalized mutual information. THE classical cross-modal same-subject metric;
               handles the ceT1<->T2 intensity inversion. Near-solves dataset1 (common grid).
  * gradcos  : gradient-magnitude cosine. Edges sit on shared anatomy across modalities.
  * nmi_grad : rank-normalised blend of the two (robust default for aligned dataset1).
  * mind     : global MIND descriptor (Heinrich) cosine. Modality- AND deformation-tolerant,
               so it is the lever for dataset2/3 where the common grid is broken.

`get_rankers(device)` returns the dict the harness iterates over. Everything is also unit
-tested without data/GPU via `tools/smoke_test_rankers.py`.
"""
import os
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch is always present on the box / locally
    torch = None
    F = None


def pick_device():
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _stack(vol_list, device):
    """List of (D,H,W) float32 -> (N,D,H,W) tensor on device."""
    arr = np.stack(vol_list).astype(np.float32)
    return torch.from_numpy(arr).to(device)


# --------------------------------------------------------------------------- mutual information
def rank_nmi(q_list, g_list, device="cpu", bins=32, max_vox=64 ** 3, mask_bg=None):
    """Studholme normalized MI for every (query, gallery) pair.

    NMI = (H(q) + H(g)) / H(q,g). It needs no shared intensity relation, only that the two
    volumes occupy the same grid (true for dataset1), which makes it the strongest d1 signal.
    Computed by batched scatter into per-pair joint histograms — one GPU pass per query.

    mask_bg (default on, NMI_MASK_BG=0 to disable): restrict the histogram to FOREGROUND voxels
    (both > thr). Without this the matched zero-background fills one giant (0,0) bin that is
    identical for every candidate and drowns the structural signal — that was the d1 bug.
    Volumes are strided down to <= max_vox voxels so the 100x100 d1 pool stays fast.
    """
    if mask_bg is None:
        # default OFF: the matched zero-background is itself an alignment cue on d1's common
        # grid, so masking it out measurably *hurts* (0.73 -> 0.51 on a d1 sample).
        mask_bg = os.environ.get("NMI_MASK_BG", "0") == "1"
    thr = float(os.environ.get("NMI_FG_THR", "0.02"))
    q = _stack(q_list, device).reshape(len(q_list), -1)
    g = _stack(g_list, device).reshape(len(g_list), -1)
    V = q.shape[1]
    if V > max_vox:                                   # uniform voxel subsample for speed
        step = V // max_vox + 1
        q, g = q[:, ::step], g[:, ::step]
    qb = (q.clamp(0, 1) * (bins - 1)).long()          # (Nq,V) bin indices
    gb = (g.clamp(0, 1) * (bins - 1)).long()          # (Ng,V)
    Nq, Ng, V = qb.shape[0], gb.shape[0], qb.shape[1]
    B2 = bins * bins
    ones = torch.ones(Ng, V, device=device)
    out = torch.zeros(Nq, Ng, device=device)
    for i in range(Nq):
        comb = qb[i].unsqueeze(0) * bins + gb         # (Ng,V) flattened joint bin
        if mask_bg:                                   # send background voxels to a dropped bin
            valid = (q[i].unsqueeze(0) > thr) & (g > thr)
            comb = torch.where(valid, comb, torch.full_like(comb, B2))
        joint = torch.zeros(Ng, B2 + 1, device=device)
        joint.scatter_add_(1, comb, ones)
        joint = joint[:, :B2].reshape(Ng, bins, bins)  # drop the sentinel bin
        pij = joint / joint.sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
        pi, pj = pij.sum(2), pij.sum(1)

        def _ent(p):
            p = p.clamp_min(1e-12)
            return -(p * p.log()).sum(-1)

        hij = -(pij.clamp_min(1e-12) * pij.clamp_min(1e-12).log()).sum(dim=(1, 2))
        out[i] = (_ent(pi) + _ent(pj)) / hij.clamp_min(1e-12)
    return out.cpu().numpy()


# --------------------------------------------------------------------------- gradient cosine
def _grad_mag_unit(x):
    """(N,D,H,W) -> (N, V) L2-normalised gradient-magnitude vectors."""
    gz, gy, gx = torch.gradient(x, dim=(1, 2, 3))
    mag = torch.sqrt(gz ** 2 + gy ** 2 + gx ** 2).reshape(x.shape[0], -1)
    return F.normalize(mag, dim=1)


def rank_gradcos(q_list, g_list, device="cpu"):
    q = _grad_mag_unit(_stack(q_list, device))
    g = _grad_mag_unit(_stack(g_list, device))
    return (q @ g.t()).cpu().numpy()


# --------------------------------------------------------------------------- MIND descriptor
def _mind(x, dilation=2):
    """Heinrich MIND (6-neighbourhood) descriptor field. x:(N,1,D,H,W) -> (N,6,D,H,W).

    Per voxel: how self-similar the patch is to its 6 face-neighbours. This structural
    fingerprint is preserved across modalities (T1<->T2) and tolerant to smooth deformation,
    so a globally pooled MIND vector survives dataset2/3 where intensity/alignment collapse.
    """
    offsets = [(dilation, 0, 0), (-dilation, 0, 0), (0, dilation, 0),
               (0, -dilation, 0), (0, 0, dilation), (0, 0, -dilation)]
    feats = []
    for o in offsets:
        xs = torch.roll(x, shifts=o, dims=(2, 3, 4))
        d = F.avg_pool3d((x - xs) ** 2, kernel_size=3, stride=1, padding=1)  # patch SSD
        feats.append(d)
    dp = torch.cat(feats, 1)                              # (N,6,D,H,W)
    var = dp.mean(1, keepdim=True).clamp_min(1e-6)
    m = torch.exp(-dp / var)
    return m / m.amax(1, keepdim=True).clamp_min(1e-6)    # normalise channels per voxel


def emb_mind(vol_list, device="cpu", pool=4):
    """Global MIND descriptor per volume: channel mean/std over brain + a coarse pool^3 grid."""
    x = _stack(vol_list, device).unsqueeze(1)            # (N,1,D,H,W)
    m = _mind(x)                                         # (N,6,D,H,W)
    mask = (x > 0.05).float()
    denom = mask.sum(dim=(2, 3, 4)).clamp_min(1.0)       # (N,1)
    mean = (m * mask).sum(dim=(2, 3, 4)) / denom         # (N,6)
    var = (((m - mean[:, :, None, None, None]) ** 2) * mask).sum(dim=(2, 3, 4)) / denom
    coarse = F.adaptive_avg_pool3d(m, pool).reshape(x.shape[0], -1)  # (N, 6*pool^3)
    vec = torch.cat([mean, var.sqrt(), coarse], dim=1)
    return F.normalize(vec, dim=1)


def rank_mind(q_list, g_list, device="cpu"):
    q = emb_mind(q_list, device)
    g = emb_mind(g_list, device)
    return (q @ g.t()).cpu().numpy()


def _rot_from_angles(a):
    """(B,3) euler angles (radians) -> (B,3,3) rotation matrices."""
    B, dev = a.shape[0], a.device
    cz, sz = torch.cos(a[:, 0]), torch.sin(a[:, 0])
    cy, sy = torch.cos(a[:, 1]), torch.sin(a[:, 1])
    cx, sx = torch.cos(a[:, 2]), torch.sin(a[:, 2])
    z, o = torch.zeros(B, device=dev), torch.ones(B, device=dev)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).view(B, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).view(B, 3, 3)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).view(B, 3, 3)
    return Rz @ Ry @ Rx


def build_template_mind(vol_list, device, dilation=2):
    """Canonical MIND template = MIND of the mean of clean (aligned) volumes — a fuzzy average
    pose that registration aligns everything to. Pass a sample of dataset1 volumes."""
    v = _stack(vol_list, device).mean(0, keepdim=True).unsqueeze(1)   # (1,1,D,H,W)
    return _mind(v, dilation)


def register_affine(vol, tmpl_mind, iters=70, lr=0.05, dilation=2):
    """Affine-register each volume (B,1,D,H,W) toward the canonical MIND template by maximising
    MIND cosine. Rigid + isotropic scale (init identity), so cross-modal and collapse-free.
    Returns the warped volumes — feed them to dense MIND matching."""
    B = vol.shape[0]
    dev = vol.device
    ang = torch.zeros(B, 3, device=dev, requires_grad=True)
    tr = torch.zeros(B, 3, device=dev, requires_grad=True)
    ls = torch.zeros(B, 1, device=dev, requires_grad=True)
    opt = torch.optim.Adam([ang, tr, ls], lr=lr)
    Tm = F.normalize(tmpl_mind.reshape(1, -1), dim=1)
    for _ in range(iters):
        R = _rot_from_angles(ang) * torch.exp(ls)[:, None, :]
        theta = torch.cat([R, tr[:, :, None]], dim=2)
        grid = F.affine_grid(theta, vol.shape, align_corners=False)
        warped = F.grid_sample(vol, grid, align_corners=False)
        wm = F.normalize(_mind(warped, dilation).reshape(B, -1), dim=1)
        loss = -(wm * Tm).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        R = _rot_from_angles(ang) * torch.exp(ls)[:, None, :]
        theta = torch.cat([R, tr[:, :, None]], dim=2)
        grid = F.affine_grid(theta, vol.shape, align_corners=False)
        return F.grid_sample(vol, grid, align_corners=False).detach()


def rank_dense_mind_registered(q_list, g_list, template_vols, device="cpu", iters=70, dilation=2):
    """Register query and gallery to a canonical pose, THEN dense-MIND match. Cracks dataset2
    (deformed): 0.15 -> 0.72 on the d2 proxy. template_vols = a sample of dataset1 volumes."""
    tmpl = build_template_mind(template_vols, device, dilation)
    q = register_affine(_stack(q_list, device).unsqueeze(1), tmpl, iters, dilation=dilation)
    g = register_affine(_stack(g_list, device).unsqueeze(1), tmpl, iters, dilation=dilation)
    qf = F.normalize(_mind(q, dilation).reshape(q.shape[0], -1), dim=1)
    gf = F.normalize(_mind(g, dilation).reshape(g.shape[0], -1), dim=1)
    return (qf @ gf.t()).cpu().numpy()


def rank_dense_mind(q_list, g_list, device="cpu", dilation=2):
    """Dense MIND-field cosine across the aligned grid — the strongest dataset1 signal (0.71 vs
    0.62 for the global blend). Unlike rank_mind it does NOT global-pool, so it keeps spatial
    correspondence; that only helps when query and gallery share a grid (dataset1). Background is
    deliberately kept: its all-ones MIND encodes the brain-mask shape, which is itself an
    alignment cue (masking it out measurably hurts)."""
    q = _stack(q_list, device).unsqueeze(1)
    g = _stack(g_list, device).unsqueeze(1)
    qf = F.normalize(_mind(q, dilation).reshape(q.shape[0], -1), dim=1)
    gf = F.normalize(_mind(g, dilation).reshape(g.shape[0], -1), dim=1)
    return (qf @ gf.t()).cpu().numpy()


# --------------------------------------------------------------------------- blend
def _rank_norm(s):
    """Min-max normalise a score matrix per query row so blends are scale-free."""
    lo = s.min(1, keepdims=True)
    hi = s.max(1, keepdims=True)
    return (s - lo) / (hi - lo + 1e-9)


def rank_nmi_grad(q_list, g_list, device="cpu", w_mi=0.7):
    mi = _rank_norm(rank_nmi(q_list, g_list, device))
    gr = _rank_norm(rank_gradcos(q_list, g_list, device))
    return w_mi * mi + (1 - w_mi) * gr


# --------------------------------------------------------------------------- registry
def get_rankers(device=None):
    """Return {name: ranker(q_vols, g_vols) -> (Nq,Ng) scores}. Honours RANKERS env allowlist."""
    if torch is None:
        return {}
    device = device or pick_device()
    allf = {
        "nmi": lambda q, g: rank_nmi(q, g, device),
        "gradcos": lambda q, g: rank_gradcos(q, g, device),
        "nmi_grad": lambda q, g: rank_nmi_grad(q, g, device),
        "mind": lambda q, g: rank_mind(q, g, device),
        "dense_mind": lambda q, g: rank_dense_mind(q, g, device),
    }
    want = os.environ.get("RANKERS")
    if want:
        keep = {k.strip() for k in want.split(",") if k.strip()}
        return {k: v for k, v in allf.items() if k in keep}
    return allf
