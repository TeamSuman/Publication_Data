"""Select 100 sparsely placed PathGennie configurations per fibril tip in the
Q_fibril / fraction-of-total-contacts projection and write them as structures.

Selection is farthest-point sampling (greedy max-min) on the 2-D projection,
which spreads the 100 points as evenly as possible over the occupied region and
therefore also reaches the sparsely populated corners of the plane.  It is
deterministic: the first point is the frame of lowest Q_fibril.
"""
import os
import numpy as np
import MDAnalysis as mda
import warnings
warnings.filterwarnings("ignore")

# Root of the PathGennie run directory, which holds U-shape/ and S-shape/.  Those
# trajectories are not part of this package; point PATHGENNIE_ROOT at your own copy.
PG = os.environ.get("PATHGENNIE_ROOT", "PathGennie")
# CV arrays written by pathgennie_cvs.py, and where the selected structures go.
CVS = os.environ.get("PATHGENNIE_CV_DIR", "cvs")
DATA = os.environ.get("SEED_OUT_DIR", "../structures/pathgennie_seeds")

N_SEEDS = 100

FOLDS = {
    "nu_8EZD": dict(
        gro=f"{PG}/U-shape/Unbinding/npt.gro",
        pep="index 0-466 and name CA",
        chains=["A", "B", "C", "D"],          # box order; see data/README.md
        roles=["incoming peptide", "growth end", "restrained", "outer"],
        trajs={"unbind": f"{PG}/U-shape/Unbinding/3D_CV/Unbind_Path_merged2.xtc",
               "bind":   f"{PG}/U-shape/Binding/Bind_Path_merged_new.xtc"},
        label="nu-shaped (8EZD-derived)",
    ),
    "S_7Q4M": dict(
        gro=f"{PG}/S-shape/Unbinding/npt.gro",
        pep="index 1401-1867 and name CA",
        chains=["A", "C", "E", "G"],
        roles=["growth end", "restrained", "outer", "incoming peptide"],
        trajs={"unbind": f"{PG}/S-shape/Unbinding/unbinding_traj_S_merged.xtc",
               "bind":   f"{PG}/S-shape/Binding/Binding_taj_S_merged.xtc"},
        label="S-shaped (7Q4M-derived)",
    ),
}


def farthest_point_sample(X, k):
    """Greedy max-min sampling; deterministic, starts at lowest y (Q_fibril)."""
    start = int(np.argmin(X[:, 1]))
    sel = [start]
    d2 = ((X - X[start]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(np.argmax(d2))
        sel.append(i)
        d2 = np.minimum(d2, ((X - X[i]) ** 2).sum(1))
    return np.array(sel)


def main():
    summary = []
    for fold, cfg in FOLDS.items():
        parts, src, frm = [], [], []
        for tag in ("unbind", "bind"):
            a = np.load(f"{CVS}/{fold}_{tag}.npy")
            parts.append(a)
            src += [tag] * len(a)
            frm += list(range(len(a)))
        cv = np.vstack(parts)
        src = np.array(src)
        frm = np.array(frm)

        all_contact = cv[:, 2]
        cmax = all_contact.max()
        x = all_contact / cmax              # fraction of total contacts, 0-1
        y = cv[:, 3]                        # Q_fibril
        X = np.column_stack([x, y])

        sel = farthest_point_sample(X, N_SEEDS)
        sel = sel[np.argsort(y[sel])]       # name in order of increasing Q_fibril

        outdir = f"{DATA}/{fold}"
        os.makedirs(outdir, exist_ok=True)

        # group the selected frames by source trajectory so each is opened once
        prefix = "nu" if fold.startswith("nu") else "S"
        rows = []
        universes = {}
        for tag, path in cfg["trajs"].items():
            u = mda.Universe(cfg["gro"], path)
            if not hasattr(u.atoms, "chainIDs"):
                u.add_TopologyAttr("chainIDs")
            universes[tag] = u
            prot = u.select_atoms("protein")
            # chain ids in box order, one per chain (see data/README.md)
            res = prot.residues
            nres = len(res) // 4
            assert len(res) % 4 == 0, f"{fold}: {len(res)} residues not divisible by 4"
            for ci, cid in enumerate(cfg["chains"]):
                res[ci * nres:(ci + 1) * nres].atoms.chainIDs = cid

        for rank, idx in enumerate(sel, start=1):
            tag, k = src[idx], int(frm[idx])
            u = universes[tag]
            u.trajectory[k]
            prot = u.select_atoms("protein")
            name = f"{prefix}_seed_{rank:03d}.pdb"
            prot.write(f"{outdir}/{name}")
            rows.append((name, tag, k, y[idx], x[idx], all_contact[idx],
                         cv[idx, 1], cv[idx, 0]))

        with open(f"{outdir}/seeds_manifest.csv", "w") as fh:
            fh.write("structure,source_ensemble,frame_in_merged_trajectory,"
                     "Q_fibril,fraction_total_contacts,N_soft_all,N_soft_native,"
                     "min_interchain_distance_nm\n")
            for r in rows:
                fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.6f},{r[4]:.6f},"
                         f"{r[5]:.4f},{r[6]:.4f},{r[7]:.4f}\n")

        np.save(f"{CVS}/{fold}_projection.npy", np.column_stack([x, y]))
        np.save(f"{CVS}/{fold}_selected.npy", sel)
        nb = (src[sel] == "bind").sum()
        summary.append(dict(fold=fold, n=len(cv), cmax=cmax, sel=len(sel),
                            nb=nb, nu_=len(sel) - nb,
                            qr=(y[sel].min(), y[sel].max()),
                            xr=(x[sel].min(), x[sel].max())))
        print(f"[{fold}] pool {len(cv)} frames  N_soft_all max {cmax:.2f}  "
              f"wrote {len(sel)} pdb -> {outdir}", flush=True)
        print(f"   from bind {nb} / unbind {len(sel)-nb};  "
              f"Q_fibril {y[sel].min():.3f}-{y[sel].max():.3f};  "
              f"frac contacts {x[sel].min():.3f}-{x[sel].max():.3f}", flush=True)

    return summary


if __name__ == "__main__":
    main()
