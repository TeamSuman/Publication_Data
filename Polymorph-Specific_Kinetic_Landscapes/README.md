# The Fibril Tip Sets the Route: Polymorph-Specific Kinetic Landscapes for Aβ42 Monomer Incorporation

Repository for "The Fibril Tip Sets the Route: Polymorph-Specific Kinetic Landscapes for Aβ42 Monomer
Incorporation".

## Overview

This repository contains the notebooks, scripts and supporting simulation assets used to compare monomer
incorporation at two structurally distinct, brain-derived Aβ42 fibril tips. The workflow combines adaptive
unbiased molecular dynamics, a trimodal VAMPnet state decomposition, a hydrogen-bond classifier of the
growing interface, Markov state modelling, and reactive-flux and committor analysis of the incorporation
pathways.

The included systems cover:

- The nu-shaped type-A fold (PDB 8EZD), four chains, with an incoming Abeta(12-42) peptide at the growth end.
- The S-shaped type-II filament (PDB 7Q4M), in the same four-chain setup with the identical peptide.

Both tips are treated identically throughout, so any difference in the kinetics follows from the template
rather than from the peptide.

## Notebooks

The `code` directory contains the notebooks used to reproduce the analysis and figures:

- `01_Trimodal_VAMPnet.ipynb`: construction of the trimodal input, training call, and inference with the
  saved models. Needs the feature files, which are not included; not executed here.
- `02_HBond_49_and_11_States.ipynb`: per-region hydrogen-bond states, coverage of the 49 microstates, the
  49 to 11 macrostate map, and the 11-state occupancies.
- `03_First_Engagement_from_Melted.ipynb`: first engagement on leaving the Melted state, the dwell-filter
  scan, landing states, and the trajectory bootstrap.
- `04_MSM_and_Pathways.ipynb`: implied timescales, the Chapman-Kolmogorov test, the MSM at 5 ns, stationary
  weights, residence times, MFPTs, committors, and the reactive-flux pathways.

Notebooks 02-04 run from the files in this repository (`cd code` first). The hydrogen-bond counts are
loaded.

## Scripts

`code` also contains the reusable Python utilities used by the notebooks:

- `abeta_tips.py`: fold settings, loading of the saved arrays, the per-region hydrogen-bond states, the 49
  microstates and the 49 to 11 macrostate rule.
- `trimodal_vampnet.py`: distance and dihedral featurization, assembly of the three input blocks, the
  network, training and inference.
- `pathgennie_cvs.py`: per-frame `Q_fibril`, soft native and total contact numbers, and minimum interchain
  distance for the merged PathGennie binding and unbinding ensembles. The `Q_fibril` double sum is
  vectorized and agrees with the original loop to 7e-16; the reference is always the fully incorporated
  `npt.gro` structure.
- `pathgennie_seed_selection.py`: farthest-point sampling of 100 configurations per tip in that projection,
  and writing of `structures/pathgennie_seeds` with its manifest.

The two `pathgennie_*` scripts need the PathGennie trajectories.

## Assets

The remaining directories contain the supporting data required by the notebooks:

- `structures/`: four-chain Abeta(12-42) constructs after N-terminal ACE capping, the input to
  `gmx pdb2gmx` (`nu_8EZD_capped_tetramer.pdb`, `S_7Q4M_capped_tetramer.pdb`). `pathgennie_seeds/<fold>/`
  holds 100 PathGennie configurations per tip, spread over the `Q_fibril` and fraction-of-total-contacts
  projection by farthest-point sampling and numbered by increasing `Q_fibril`, with `seeds_manifest.csv`
  giving the source ensemble, the frame index and the CV values of each structure. This is a sparse,
  evenly spaced sample of the pathway-generation output, not the 1,988 and 1,026 clustered seeds that
  were actually run.
- `topology/<fold>/`: `topol.top`, one `topol_Protein_chain_*.itp` per chain, and
  `posre_Protein_chain_C.itp` (heavy atoms of the restrained chain, 1000 kJ mol^-1 nm^-2, activated with
  `define = -DPOSRES_C`). AMBER99SB-ILDN with TIP3P water.
- `representative_structures/kinetic_states/<fold>/`: for each of the six kinetic states, the template
  (`sN_fibril.pdb`), the six peptide conformations drawn in the route-map figure (`sN_pep0-5.pdb`) and the
  state medoid (`sN_medoid.pdb`).
- `representative_structures/11_state_classes/<fold>/`: the medoid structure of each of the 11
  hydrogen-bond macrostates (`state_mNN_<name>.pdb`, where NN is the macrostate index of the SI Tier-2
  table).
- `vampnet_models/<fold>/`: the trained trimodal VAMPnet weights
  (`trimodal_vampnet_lobe_state_dict.pt`), the distance and dihedral scaler, the hydrogen-bond bitmask
  columns (`top_hbond_indices.npy`, `top_hbond_pairs.npy`), and the training and validation VAMP-2 curves.
- `saved_data/<fold>/`: the six-state assignment of every frame (`trimodal_state_assignments.npy`) and the
  per-frame native and non-native interfacial hydrogen-bond counts of beta1 and beta2
  (`Native_Hbond_beta{1,2}.npy`, `Non-Native_Hbond_beta{1,2}.npy`).

Chain roles, as 0-based atom indices:

| | incoming peptide | growth end | restrained | outer |
|---|---|---|---|---|
| nu_8EZD | A (0-466) | B (467-933) | C (934-1400) | D (1401-1867) |
| S_7Q4M | G (1401-1867) | A (0-466) | C (467-933) | E (934-1400) |

Trajectories are stored every 0.1 ns. The frame arrays in `saved_data/` are all trajectories of a tip
concatenated in the order of `abeta_tips.FOLDS[fold]["batches"]`. In both systems the first round is 451
frames per trajectory and the adaptive rounds that follow it are 501 frames: 1,988 then 804 trajectories
for the nu tip (129.9 us in total) and 1,026 then 683 for the S tip (80.5 us).

## Requirements

Python 3 with numpy 1.26.4, scikit-learn 1.7.2, PyTorch 2.5.1, deeptime 0.4.5 and MDAnalysis 2.9.0. The
simulations were run with GROMACS 2022.5.
