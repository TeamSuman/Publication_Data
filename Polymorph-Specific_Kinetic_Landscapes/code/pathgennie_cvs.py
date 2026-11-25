"""Compute PathGennie projection CVs (Q_fibril vs all-contact) for the merged
binding and unbinding ensembles of both fibril-tip systems.

CV definitions are taken verbatim from Path_gennie_analysis.ipynb
(projection_qfibril / projection_qfibril_Bind), with the Q_fibril double loop
vectorized.  The Q_fibril reference is the npt.gro fully-incorporated structure,
as in projection_qfibril_Bind; for the unbinding paths frame 0 is that same
structure, so this is consistent across both ensembles.

Output columns per frame: [min_dist_nm, native_contact, all_contact, q_fibril]
"""
import os, sys, time, warnings
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import distances

warnings.filterwarnings("ignore")

# Root of the PathGennie run directory, which holds U-shape/ and S-shape/.  Those
# trajectories are not part of this package; point PATHGENNIE_ROOT at your own copy.
PG = os.environ.get("PATHGENNIE_ROOT", "PathGennie")
# Where the per-frame CV arrays are written (read back by pathgennie_seed_selection.py).
OUT = os.environ.get("PATHGENNIE_CV_DIR", "cvs")

R_0, N_EXP, M_EXP, D_0 = 5.0, 6, 12, 2.0
SIGMA_0, EPSILON = 1.0, 0.15

FOLDS = {
    "nu_8EZD": dict(
        gro=f"{PG}/U-shape/Unbinding/npt.gro",
        pep="index 0-466 and name CA",
        tip="index 467-933 and name CA",
        trajs=[("unbind", f"{PG}/U-shape/Unbinding/3D_CV/Unbind_Path_merged2.xtc"),
               ("bind",   f"{PG}/U-shape/Binding/Bind_Path_merged_new.xtc")],
    ),
    "S_7Q4M": dict(
        gro=f"{PG}/S-shape/Unbinding/npt.gro",
        pep="index 1401-1867 and name CA",
        tip="index 0-466 and name CA",
        trajs=[("unbind", f"{PG}/S-shape/Unbinding/unbinding_traj_S_merged.xtc"),
               ("bind",   f"{PG}/S-shape/Binding/Binding_taj_S_merged.xtc")],
    ),
}


def switch_sum(d, diagonal_only=False):
    """Soft contact count with the notebook's switching function."""
    if diagonal_only:
        d = np.diag(d)
    ratio = (d - D_0) / R_0
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (1 - ratio ** N_EXP) / (1 - ratio ** M_EXP)
        s[ratio > 1.0] = 0.0
    return np.nansum(s)


def q_fibril_loop(r_k, r_l, n_pairs):
    """Original double-loop implementation, for validation only."""
    tot = 0.0
    for i in range(r_k.shape[0]):
        for j in range(r_k.shape[1]):
            if i != j:
                sig = SIGMA_0 * abs(i - j) ** EPSILON
                tot += np.exp(-((r_k[i, j] - r_l[i, j]) ** 2) / (sig ** 2))
    return tot / n_pairs


def make_sigma2(n):
    idx = np.arange(n)
    sig = SIGMA_0 * np.abs(idx[:, None] - idx[None, :]).astype(float) ** EPSILON
    np.fill_diagonal(sig, 1.0)          # placeholder, diagonal is dropped below
    return sig ** 2


def q_fibril_vec(r_k, r_l, sigma2, n_pairs):
    q = np.exp(-((r_k - r_l) ** 2) / sigma2)
    np.fill_diagonal(q, 0.0)
    return q.sum() / n_pairs


def run_fold(fold, validate=False):
    cfg = FOLDS[fold]
    ref = mda.Universe(cfg["gro"])
    ref_pep = ref.select_atoms(cfg["pep"])
    r_l = distances.distance_array(ref_pep.positions.copy(),
                                   ref_pep.positions.copy(), box=ref.dimensions)
    n = r_l.shape[0]
    n_pairs = n * (n - 1)
    sigma2 = make_sigma2(n)
    print(f"[{fold}] reference {cfg['gro'].split('/')[-1]}  peptide CA = {n}", flush=True)

    for tag, xtc in cfg["trajs"]:
        u = mda.Universe(cfg["gro"], xtc)
        pep = u.select_atoms(cfg["pep"])
        tip = u.select_atoms(cfg["tip"])
        nfr = len(u.trajectory)
        print(f"[{fold}/{tag}] {nfr} frames  ({xtc.split('/')[-1]})", flush=True)

        out = np.empty((nfr, 4), dtype=np.float64)
        t0 = time.time()
        for k, ts in enumerate(u.trajectory):
            dv = distances.distance_array(pep.positions, tip.positions, box=u.dimensions)
            out[k, 0] = dv.min() / 10.0
            out[k, 1] = switch_sum(dv, diagonal_only=True)
            out[k, 2] = switch_sum(dv)
            r_k = distances.distance_array(pep.positions, pep.positions, box=u.dimensions)
            out[k, 3] = q_fibril_vec(r_k, r_l, sigma2, n_pairs)

            if validate and k < 2:
                ref_q = q_fibril_loop(r_k, r_l, n_pairs)
                print(f"   validate frame {k}: vec={out[k,3]:.10f} loop={ref_q:.10f} "
                      f"diff={abs(out[k,3]-ref_q):.3e}", flush=True)
            if k % 2000 == 0 and k:
                el = time.time() - t0
                print(f"   {k}/{nfr}  {el:.0f}s  eta {el/k*(nfr-k):.0f}s", flush=True)
            if validate and k >= 2:
                return

        np.save(f"{OUT}/{fold}_{tag}.npy", out)
        print(f"[{fold}/{tag}] saved {out.shape} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    validate = "--validate" in sys.argv
    folds = [a for a in sys.argv[1:] if a in FOLDS] or list(FOLDS)
    for f in folds:
        run_fold(f, validate=validate)
