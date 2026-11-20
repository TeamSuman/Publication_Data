"""Fold settings, data loading and the H-bond state classifier (49 microstates -> 11 macrostates).

Two four-chain Abeta(12-42) fibril tips:
    nu_8EZD  (nu fold)   chain A incoming peptide, B growth end, C restrained, D outer
    S_7Q4M   (S fold)    chain G incoming peptide, A growth end, C restrained, E outer

Frames are stored every 0.1 ns. The first 1,988 (nu) / 1,026 (S) trajectories are the
PathGennie-seeded runs (451 frames each, first 5 ns removed); the rest are the adaptive
rounds (501 frames each).
"""
import os
import numpy as np

DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT_NS = 0.1

MACRO_NAMES = ["Melted", "Pure Native", "b1 Anchored", "b2 Anchored", "Misregistered Trap",
               "Frustrated Trap", "b1 Frustrated", "b2 Frustrated", "b2 Exploring",
               "b1 Exploring", "Native Defective"]
MELTED = 0
# grouped by the strand that is engaged (the Frustrated label names the flickering strand)
B1_ENGAGED = {2, 9, 7}
B2_ENGAGED = {3, 8, 6}

REGION_STATES = ["Melted", "Small/Small", "Small/Large", "Large/Small", "Large/Large",
                 "Pure Non-Native", "Pure Native"]


def _cuts_nu(n1, nn1, n2, nn2):
    b1 = [(n1 == 0) & (nn1 == 0),
          (n1 >= 1) & (n1 < 4) & (nn1 == 1),
          (n1 >= 1) & (n1 < 4) & (nn1 >= 2),
          (n1 >= 4) & (nn1 == 1),
          (n1 >= 4) & (nn1 >= 2),
          (n1 == 0) & (nn1 > 0),
          (n1 > 0) & (nn1 == 0)]
    b2 = [(n2 == 0) & (nn2 == 0),
          (n2 >= 1) & (n2 < 4) & (nn2 >= 1) & (nn2 < 3),
          (n2 >= 1) & (n2 < 4) & (nn2 >= 3),
          (n2 >= 4) & (nn2 >= 1) & (nn2 < 3),
          (n2 >= 4) & (nn2 >= 3),
          (n2 == 0) & (nn2 > 0),
          (n2 > 0) & (nn2 == 0)]
    return np.select(b1, range(7), default=-1), np.select(b2, range(7), default=-1)


def _cuts_S(n1, nn1, n2, nn2):
    b1 = [(n1 == 0) & (nn1 == 0),
          (n1 == 1) & (nn1 >= 1) & (nn1 <= 2),
          (n1 == 1) & (nn1 >= 3),
          (n1 >= 2) & (nn1 >= 1) & (nn1 <= 2),
          (n1 >= 2) & (nn1 >= 3),
          (n1 == 0) & (nn1 > 0),
          (n1 > 0) & (nn1 == 0)]
    b2 = [(n2 == 0) & (nn2 == 0),
          (n2 >= 1) & (n2 < 5) & (nn2 >= 1) & (nn2 <= 2),
          (n2 >= 1) & (n2 < 5) & (nn2 >= 3),
          (n2 >= 5) & (nn2 >= 1) & (nn2 <= 2),
          (n2 >= 5) & (nn2 >= 3),
          (n2 == 0) & (nn2 > 0),
          (n2 > 0) & (nn2 == 0)]
    return np.select(b1, range(7), default=-1), np.select(b2, range(7), default=-1)


FOLDS = {
    "nu_8EZD": dict(
        batches=[(1988, 451), (804, 501)],
        source=5, sink=2,
        n_hbond=86,
        peptide_ca="index 0-466 and name CA",
        growth_ca="index 467-933 and name CA",
        dihedral_sel="index 0-933",
        cuts=_cuts_nu,
    ),
    "S_7Q4M": dict(
        batches=[(1026, 451), (683, 501)],
        source=1, sink=5,
        n_hbond=75,
        peptide_ca="index 1401-1867 and name CA",
        growth_ca="index 0-466 and name CA",
        dihedral_sel="index 1401-1867 or index 0-466",
        cuts=_cuts_S,
    ),
}


def traj_lengths(fold):
    return np.concatenate([np.full(n, L) for n, L in FOLDS[fold]["batches"]])


def split_per_traj(x, fold):
    return np.split(np.asarray(x), np.cumsum(traj_lengths(fold))[:-1])


def load_dtrajs(fold):
    """Six-state VAMPnet assignment of every frame, one array per trajectory."""
    sa = np.load(os.path.join(DATA, "saved_data", fold, "trimodal_state_assignments.npy"))
    assert len(sa) == traj_lengths(fold).sum()
    return [t.astype(np.int32) for t in split_per_traj(sa, fold)]


def load_hbond_counts(fold):
    """Native / non-native interfacial backbone H-bond counts per frame (beta1 16-22, beta2 30-40).

    These are the stored output of the H-bond pass (MDAnalysis HydrogenBondAnalysis,
    3.5 A donor-acceptor, 150 deg D-H-A, incoming peptide vs growth end).
    """
    d = os.path.join(DATA, "saved_data", fold)
    n1 = np.load(os.path.join(d, "Native_Hbond_beta1.npy"))
    nn1 = np.load(os.path.join(d, "Non-Native_Hbond_beta1.npy"))
    n2 = np.load(os.path.join(d, "Native_Hbond_beta2.npy"))
    nn2 = np.load(os.path.join(d, "Non-Native_Hbond_beta2.npy"))
    return n1, nn1, n2, nn2


def region_states(fold, counts=None):
    """Per-region state (0-6) of beta1 and beta2 in every frame."""
    counts = load_hbond_counts(fold) if counts is None else counts
    s1, s2 = FOLDS[fold]["cuts"](*counts)
    assert (s1 >= 0).all() and (s2 >= 0).all()
    return s1, s2


def microstates(s1, s2):
    """Joint microstate index 0-48 (= 7 * beta1 state + beta2 state)."""
    return 7 * s1 + s2


# tier 1: seven region states -> five tokens
_TOKEN = {0: "M", 1: "F", 2: "NN", 3: "N_def", 4: "F", 5: "NN", 6: "N_pure"}


def _tier2(c1, c2):
    anch = ("N_pure", "N_def")
    if c1 == "M" and c2 == "M":
        return 0
    if c1 == "N_pure" and c2 == "N_pure":
        return 1
    if c1 in anch and c2 in anch:
        return 10
    if c1 in anch and c2 in ("M", "NN"):
        return 2
    if c2 in anch and c1 in ("M", "NN"):
        return 3
    if c1 == "F" and c2 in anch:
        return 6
    if c2 == "F" and c1 in anch:
        return 7
    if c1 == "NN" and c2 == "NN":
        return 4
    if c1 in ("F", "NN") and c2 in ("F", "NN"):
        return 5
    if c1 == "M" and c2 in ("F", "NN"):
        return 8
    if c2 == "M" and c1 in ("F", "NN"):
        return 9
    raise ValueError(f"unmapped pair {c1}, {c2}")


MICRO_TO_MACRO = np.array([_tier2(_TOKEN[a], _TOKEN[b]) for a in range(7) for b in range(7)])


def macrostates(fold):
    """11-state label of every frame."""
    s1, s2 = region_states(fold)
    return MICRO_TO_MACRO[microstates(s1, s2)].astype(np.int8)
