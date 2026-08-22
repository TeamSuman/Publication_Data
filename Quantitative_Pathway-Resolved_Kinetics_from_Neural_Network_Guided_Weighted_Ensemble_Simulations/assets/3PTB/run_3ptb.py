#!/usr/bin/env python3
import warnings
import joblib
import numpy as np
import multiprocessing as mp
import mdtraj as md
import MDAnalysis as mda
from MDAnalysis.analysis import distances
import sys
sys.path.insert(0, "/home/suman/Dibyendu/WEPath/Scripts")
from main import WESS

warnings.filterwarnings("ignore")
def pca_projection(positions, protein_indices, ligand_indices, model):
    """
    Projects the atomic positions onto the principal components.

    This function calculates the distance features between the protein and ligand,
    and then uses a pre-trained PCA model to reduce the dimensionality.

    Args:
        positions (np.ndarray): Array of atomic positions from the simulation frame.
        **kwargs: Keyword arguments containing:
            protein_indices (np.ndarray): Indices of the protein atoms.
            ligand_indices (np.ndarray): Indices of the ligand atoms.
            model (object): The loaded PCA model object.

    Returns:
        tuple: A tuple containing:
            - The first 4 principal components of the transformed features.
            - The distance between the center of mass of the protein and ligand.
            - The minimum distance between any protein and ligand atom.
    """

    if protein_indices is None or ligand_indices is None or model is None:
        raise ValueError("Missing required arguments in projection function: protein_indices, ligand_indices, or model.")

    # Extract positions for protein and ligand
    prot_pos = positions[protein_indices]
    lig_pos = positions[ligand_indices]

    # Calculate pairwise distances and flatten to create a feature vector
    my_distance = distances.distance_array(prot_pos, lig_pos)
    feat = my_distance.ravel()

    # Calculate center of mass distance

    # Transform features using the PCA model and return key metrics
    # The original code returns the first 4 components, so we slice with [:4]
    transformed_features = model.transform(feat.reshape(1, -1))
    return transformed_features[0][:5]


def geometric_path_cv(pos, path):
    """
    Compute geometric path CV (s, z):
    s = normalized progress along path [0,1]
    z = perpendicular distance to path
    """
    diffs = path - pos
    dists = np.linalg.norm(diffs, axis=1)
    i_min = np.argmin(dists)

    if i_min == 0:
        i_next = 1
    elif i_min == len(path)-1:
        i_next = len(path)-2
    else:
        i_next = i_min + (1 if dists[i_min+1] < dists[i_min-1] else -1)

    q1, q2 = path[i_min], path[i_next]
    t = np.dot(pos - q1, q2 - q1) / np.dot(q2 - q1, q2 - q1)
    proj = q1 + t * (q2 - q1)

    s = (i_min + np.clip(t, 0, 1)) / (len(path)-1)
    z = np.linalg.norm(pos - proj)
    return s, z


def compute_bsa_from_positions(positions,
                               template,
                               ligand_sel="resname BEN",
                               protein_sel="protein",
                               probe_radius=0.14):
    """
    Compute buried surface area (BSA) between ligand and protein across frames,
    using a template PDB for topology and an array of positions.

    Args:
        template_pdb (str): Path to template PDB file (defines topology & atom ordering)
        positions (np.ndarray): Array of shape (n_frames, n_atoms, 3) in nm
                                Must match atom ordering in template_pdb
        ligand_sel (str): MDTraj selection string for ligand (default "resname BAMI")
        protein_sel (str): MDTraj selection string for protein
        probe_radius (float): probe radius in nm (default 0.14 nm = 1.4 Å)

    Returns:
        np.ndarray: BSA time series (nm^2), shape (n_frames,)
    """
    # Load topology

    # Build trajectory object with external coordinates
    traj = md.Trajectory(xyz=positions, topology=template.topology)

    # Get atom indices
    top = traj.topology
    ligand_idx = top.select(ligand_sel + " and not element H")  # heavy atoms only
    protein_idx = top.select(protein_sel)

    # SASA of ligand alone
    sasa_lig = md.shrake_rupley(traj.atom_slice(ligand_idx), probe_radius=probe_radius, mode="residue")
    sasa_lig_total = np.sum(sasa_lig, axis=1)

    # SASA of protein alone
    sasa_prot = md.shrake_rupley(traj.atom_slice(protein_idx), probe_radius=probe_radius, mode="residue")
    sasa_prot_total = np.sum(sasa_prot, axis=1)

    # SASA of protein+ligand complex
    sasa_complex = md.shrake_rupley(traj.atom_slice(np.concatenate([protein_idx, ligand_idx])),
                                    probe_radius=probe_radius, mode="residue")
    sasa_complex_total = np.sum(sasa_complex, axis=1)

    # Buried surface area
    bsa = sasa_lig_total + sasa_prot_total - sasa_complex_total
    return bsa
# -----------------------
# Warp criteria
# -----------------------
def warp_criteria(positions, kwargs):
    """Return True if contact distance is greater than 1 nm"""
    template = kwargs['template']
    bsa = compute_bsa_from_positions(positions, template)
    return bsa < 0.01


def project(positions, kwargs):
    model = kwargs['model']
    path = kwargs['path']
    protein_indices = kwargs['protein_idx']
    ligand_indices = kwargs['ligand_idx']

    pca = pca_projection(10.0*positions, protein_indices, ligand_indices, model)
    s, z = geometric_path_cv(pca, path)
    print(s, z)
    return np.array([s, z])
# -----------------------
# Main
# -----------------------
if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)

    # file paths
    ref_file = "../Data/3PTB/DRAY/md.gro"        # used by OpenMM runner 

    u = mda.Universe(ref_file)
    template = md.load(ref_file)

    u7 = mda.Universe(ref_file, '../Data/3PTB/DRAY/trajectory_awxl.xtc')

    warp_kwargs = {
        'template' : template
    }

    initial_positions = [0.1 * u.atoms.positions.copy()]
    for i in range(60):
        u = mda.Universe(f"../Data/3PTB/DRAY/Path0/frame_{i}.gro")
        initial_positions.append(0.1 * u.atoms.positions.copy())  # (n_atoms,3) in nm
    projection_fn = project


    model = joblib.load("../Data/3PTB/DRAY/pca_minmax_pipeline.pkl")
    u = mda.Universe("../Data/3PTB/DRAY/md.gro")
    # Define atom selections based on command-line arguments
    protein_selection_str = 'around 20 resname BEN'
    ligand_selection_str = 'resname BEN and not type H'
    protein_ca_indices = u.select_atoms("name CA").indices
    protein_nearby_indices = u.select_atoms(protein_selection_str).select_atoms("name CA").indices
    ligand_indices = u.select_atoms(ligand_selection_str).indices

    print(f"Found {len(protein_nearby_indices)} protein atoms and {len(ligand_indices)} ligand atoms.")

    # Prepare arguments for the projection function
    projection_kwargs = {
        "protein_idx": protein_nearby_indices,
        "ligand_idx": ligand_indices,
        "model": model,
        "path" : np.load("../Data/3PTB/DRAY/Path0/path.npy")
    }

    bin_edges = [list(np.linspace(0, 1.0, 25)), [0.0, 1.0, 4.0, np.inf]]

    config = {
        'n_gpus': 2,
        'source_bin_indices': np.array([[0]]),
        'temperature': 298.0,
        'bin_edges' :bin_edges,
        'enable_cleaning': True,
        'clean_threshold': 75.0,
        'protein_idx' : protein_ca_indices,
        'ligand_idx' : ligand_indices,
        #'bin_edges': bin_edges,
        'n_walkers_per_bin': 3,
        'dt': 0.004,
        'n_steps_per_tau': 5000,
        'n_iterations': 50000,
        'flux_file': 'flux_file_3ptb_path20ps_minor1.txt',
        'survive_empty': False,
        'warp_function': warp_criteria,
        'warp_kwargs': warp_kwargs,
    }

    we_sim = WESS(
        config=config,
        initial_positions=initial_positions,
        projection_fn=projection_fn,
        kwargs=projection_kwargs
    )
    we_sim.run()
