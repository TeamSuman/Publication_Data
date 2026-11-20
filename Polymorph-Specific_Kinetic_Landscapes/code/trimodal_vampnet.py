"""Trimodal (late-fusion) VAMPnet used for both fibril tips.

Input per frame, three blocks:
    1. 31 x 31 C-alpha distance map, incoming peptide (rows) vs growth end (cols)  -> CNN branch
    2. 244 backbone dihedral features, sin/cos of phi and psi of both chains        -> MLP branch
    3. binary bitmask of the longest-lived interfacial H-bonds (86 nu, 75 S)        -> MLP branch
Blocks 1 and 2 are min-max scaled to [-1, 1] together (one scaler over the 1205 columns,
stored as dist_dih_scaler.pkl). Each branch gives an 8-dim latent vector; the three are
concatenated, layer-normalised and classified into K = 6 states (softmax).

Training: VAMP-2, lag 5 frames (0.5 ns), Adam lr 1e-3, batch 4096, 30 epochs, random 80/20
train/validation split of the time-lagged pairs.
"""
import pickle

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

N_DIST = 961
N_DIH = 244
LATENT_DIM = 8
N_STATES = 6
LAG_TRAIN = 5


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def featurize(universe, peptide_ca, growth_ca, dihedral_sel):
    """Distance + dihedral features of one trajectory, shape (n_frames, 1205).

    Distances in Angstrom with the minimum-image convention. Dihedral atoms are made
    whole across the box before the angle is taken.
    """
    from MDAnalysis.lib.distances import distance_array, calc_dihedrals

    pep = universe.select_atoms(peptide_ca)
    grow = universe.select_atoms(growth_ca)
    res = universe.select_atoms(dihedral_sel).residues
    phi_idx = np.array([r.phi_selection().indices for r in res if r.phi_selection() is not None])
    psi_idx = np.array([r.psi_selection().indices for r in res if r.psi_selection() is not None])

    out = []
    for ts in universe.trajectory:
        box = ts.dimensions
        L = box[:3]
        d = distance_array(pep.positions, grow.positions, box=box).ravel()

        angles = []
        for idx in (phi_idx, psi_idx):
            x = ts.positions[idx].copy()
            for k in range(3):
                v = x[:, k + 1] - x[:, k]
                v -= L * np.round(v / L)
                x[:, k + 1] = x[:, k] + v
            angles.append(calc_dihedrals(x[:, 0], x[:, 1], x[:, 2], x[:, 3]))
        phi, psi = angles
        dih = np.concatenate([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)])
        out.append(np.concatenate([d, dih]))
    return np.asarray(out, dtype=np.float32)


def load_inputs(dist_dih_pkl, scaler_pkl, hbond_npz):
    """Return per-trajectory lists (dist (T,31,31), dih (T,244), hbond (T,n_hb))."""
    with open(dist_dih_pkl, "rb") as fh:
        dist_dih = pickle.load(fh)
    scaler = joblib.load(scaler_pkl)

    z = np.load(hbond_npz)
    keys = sorted(z.files, key=lambda k: int(k.split("_")[1]))
    hbond = [z[k].astype(np.float32) for k in keys]
    assert len(hbond) == len(dist_dih)

    dist, dih = [], []
    for seg in dist_dih:
        s = scaler.transform(seg).astype(np.float32)
        dist.append(s[:, :N_DIST].reshape(-1, 31, 31))
        dih.append(s[:, N_DIST:])
    return dist, dih, hbond


# --------------------------------------------------------------------------
# network
# --------------------------------------------------------------------------

class ConvEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),     # 31 -> 16
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # 16 -> 8
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # 8 -> 4
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.main(x)


class MLPEncoder(nn.Module):
    """Used for both the dihedral and the H-bond branch."""

    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 128), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ELU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class TrimodalLobe(nn.Module):
    def __init__(self, hbond_dim, n_states=N_STATES, latent_dim=LATENT_DIM):
        super().__init__()
        self.dist_encoder = ConvEncoder(latent_dim)
        self.dih_encoder = MLPEncoder(N_DIH, latent_dim)
        self.hbond_encoder = MLPEncoder(hbond_dim, latent_dim)
        self.fusion_norm = nn.LayerNorm(3 * latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * latent_dim, 64), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(64, 64), nn.ELU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ELU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ELU(),
            nn.Linear(16, n_states),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        dist, dih, hbond = x
        z = torch.cat([self.dist_encoder(dist), self.dih_encoder(dih), self.hbond_encoder(hbond)], dim=1)
        return self.classifier(self.fusion_norm(z))


def load_lobe(state_dict_pt, hbond_dim, device="cpu"):
    lobe = TrimodalLobe(hbond_dim).to(device)
    lobe.load_state_dict(torch.load(state_dict_pt, map_location=device))
    lobe.eval()
    return lobe


# --------------------------------------------------------------------------
# time-lagged pairs and training
# --------------------------------------------------------------------------

class Triple(list):
    """[dist, dih, hbond]; deeptime's VAMPNet.fit calls .to(device) on each batch."""

    def to(self, *args, **kwargs):
        return Triple([t.to(*args, **kwargs) for t in self])


class TrimodalLagged(Dataset):
    """All (t, t + lag) pairs inside each trajectory."""

    def __init__(self, dist, dih, hbond, lag=LAG_TRAIN):
        self.blocks = [(torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c))
                       for a, b, c in zip(dist, dih, hbond)]
        self.index = [(i, t) for i, blk in enumerate(self.blocks) for t in range(len(blk[0]) - lag)]
        self.lag = lag

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        i, t = self.index[k]
        d, a, h = self.blocks[i]
        s = t + self.lag
        return Triple([d[t], a[t], h[t]]), Triple([d[s], a[s], h[s]])


def train(dist, dih, hbond, device, n_epochs=30, batch_size=4096, lr=1e-3):
    from deeptime.decomposition.deep import VAMPNet

    data = TrimodalLagged(dist, dih, hbond)
    n_val = int(0.2 * len(data))
    tr, va = random_split(data, [len(data) - n_val, n_val])
    loader_tr = DataLoader(tr, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(va, batch_size=batch_size, shuffle=False)

    lobe = TrimodalLobe(hbond_dim=hbond[0].shape[1]).to(device)
    vnet = VAMPNet(lobe=lobe, learning_rate=lr, device=device)
    vnet.fit(loader_tr, n_epochs=n_epochs, validation_loader=loader_va)
    return lobe, np.asarray(vnet.train_scores), np.asarray(vnet.validation_scores)


@torch.no_grad()
def predict(lobe, dist, dih, hbond, device="cpu", batch_size=4096):
    """Soft state probabilities for every frame, trajectories concatenated in order."""
    lobe.eval()
    D = torch.from_numpy(np.concatenate(dist))
    A = torch.from_numpy(np.concatenate(dih))
    H = torch.from_numpy(np.concatenate(hbond))
    out = []
    for i in range(0, len(D), batch_size):
        x = [D[i:i + batch_size].to(device), A[i:i + batch_size].to(device), H[i:i + batch_size].to(device)]
        out.append(lobe(x).cpu())
    return torch.cat(out).numpy()
