# Data and code: Abeta(12-42) elongation at the nu (8EZD) and S (7Q4M) fibril tips

## Contents

| Folder | Contents |
|---|---|
| `structures/` | Four-chain Abeta(12-42) constructs after N-terminal ACE capping (input to `gmx pdb2gmx`): `nu_8EZD_capped_tetramer.pdb`, `S_7Q4M_capped_tetramer.pdb` |
| `topology/<fold>/` | `topol.top`, one `topol_Protein_chain_*.itp` per chain, and `posre_Protein_chain_C.itp` (heavy atoms of the restrained chain, 1000 kJ mol^-1 nm^-2; activated with `define = -DPOSRES_C`). AMBER99SB-ILDN, TIP3P. |
| `representative_structures/kinetic_states/<fold>/` | For each of the six kinetic states: `sN_fibril.pdb` (template), `sN_pep0-5.pdb` (the six peptide conformations drawn in the route-map figure), `sN_medoid.pdb` (the state medoid) |
| `representative_structures/11_state_classes/<fold>/` | Medoid structure of each of the 11 H-bond macrostates (`state_mNN_<name>.pdb`; NN is the macrostate index of the SI Tier-2 table) |
| `vampnet_models/<fold>/` | Trained trimodal VAMPnet weights (`trimodal_vampnet_lobe_state_dict.pt`), the distance+dihedral scaler, the H-bond bitmask columns (`top_hbond_indices.npy`, `top_hbond_pairs.npy`), and the training/validation VAMP-2 curves |
| `saved_data/<fold>/` | Six-state assignment of every frame (`trimodal_state_assignments.npy`), and the per-frame native/non-native interfacial H-bond counts of beta1 and beta2 (`Native_Hbond_beta{1,2}.npy`, `Non-Native_Hbond_beta{1,2}.npy`) |
| `code/` | Analysis code (below) |

Chain roles (0-based atom indices in the simulation box):

| | incoming peptide | growth end | restrained | outer |
|---|---|---|---|---|
| nu_8EZD | A (0-466) | B (467-933) | C (934-1400) | D (1401-1867) |
| S_7Q4M | G (1401-1867) | A (0-466) | C (467-933) | E (934-1400) |

Trajectories are stored every 0.1 ns. The frame arrays in `saved_data/` follow the trajectory order of
`abeta_tips.FOLDS[fold]["batches"]`: 1,988 + 804 (nu) and 1,026 + 683 (S) trajectories of 451 and 501 frames.

## Code

| File | What it does |
|---|---|
| `abeta_tips.py` | Fold settings, loading of the saved arrays, the per-region H-bond states, 49 microstates and the 49 -> 11 macrostate rule |
| `trimodal_vampnet.py` | Distance/dihedral featurization, assembly of the three input blocks, the network, training and inference |
| `01_Trimodal_VAMPnet.ipynb` | How the trimodal input is built and fed to the network; training call; inference with the saved models (needs the feature files, not included; not executed here) |
| `02_HBond_49_and_11_States.ipynb` | Region states, 49-microstate coverage, the 49 -> 11 map, 11-state occupancies |
| `03_First_Engagement_from_Melted.ipynb` | First engagement on leaving Melted: dwell-filter scan, landing states, trajectory bootstrap |
| `04_MSM_and_Pathways.ipynb` | Implied timescales, Chapman-Kolmogorov test, MSM at 5 ns, stationary weights, residence times, MFPTs, committors, reactive-flux pathways |

Notebooks 02-04 run from the files in this folder (`cd code` first). The H-bond counts are loaded, not
recomputed.

Versions used: Python 3, numpy 1.26.4, scikit-learn 1.7.2, PyTorch 2.5.1, deeptime 0.4.5,
MDAnalysis 2.9.0. Simulations: GROMACS 2022.5.
