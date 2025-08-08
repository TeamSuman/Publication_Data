from pathlib import Path
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import MDAnalysis as mda

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from joblib import dump, load
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm.auto as tqdm
from featuredistance import featuredistance
from featurecoordinate import featurecoordinates


class Feature:
    def __init__(self, scaler_file = None, model_file  = None):
            
        self.scaler_file = scaler_file
        self.model_file = model_file
        self.contacts = None
        self.atom_index = None
        
    def prepare_contacts(self, contacts_file = None, mode = "distance"):
        
        self.mode = mode
        
        if self.mode == "distance":
            if self.contacts is None:
                assert Path(contacts_file).exists(), f'{contacts_file} not exists! Specify contacts list file!'
                self.contacts = np.loadtxt(contacts_file, dtype=int)
                #TODO: check for dimensionality
                unique_atoms = np.unique(self.contacts.ravel())
                #print(self.contacts)
            else:
                unique_atoms = np.unique(self.contacts.ravel())
                
        
        elif self.mode == "coordinate":
            if self.atom_index is None:
                assert Path(contacts_file).exists(), f'{contacts_file} not exists! Specify contacts list file!'
                self.atoms = np.loadtxt(contacts_file, dtype=int)
            else:
                self.atoms = self.atom_index
                #TODO: check for dimensionality
                unique_atoms = np.unique(self.atoms)
        else:
            unique_atoms = None    
        
        self.selection = "id " + " or id ".join(np.array(unique_atoms, dtype = str))
        
        #assert Path(self.traj).exists(), f'{self.traj} not exists! Specify trajectory file (xtc/trr/dcd/nc)!'
    
    def short_fd(self, cpu_num = 0, chunk_size = 0, last_cpu = 31, step = 1, fd = None, verbose = False):
        if cpu_num == last_cpu:
            fd.run(start = cpu_num*chunk_size, step = step, verbose=verbose)
        else:
            fd.run(start = cpu_num*chunk_size, stop = (cpu_num+1)*chunk_size, step = step, verbose=verbose)
        return fd.results

    ####### Feature extraction ... ########
    def feature_extraction(self, top = None , traj = None, step = 1, frames = None, verbose = True, mda_universe = None):
        if mda_universe is None:
            self.top = top 
            self.traj = traj
            self.frames  = frames

            assert Path(self.top).exists(), f'{self.top} not exists! Specify topology(gro/pdb/tpr/parm7)!'

            if traj:
                if type(traj) == str:
                    universe =  mda.Universe(top, traj)
                elif type(traj) == list:
                    universe =  mda.Universe(top, list(traj))
            else:
                universe =  mda.Universe(top) 
        else:
            universe = mda_universe

        atomgroup = universe.select_atoms(self.selection)
        
        #print(printColor.BLUE + printColor.BOLD + "Extracting features from trajectory.." + printColor.END)
        #print(self.mode)
        ## TODO : why indexing occurs
        if self.mode == "distance":
            #print("Distance based features..\n")
            fd = featuredistance(atomgroup, self.contacts)
            #n_jobs = -1
            #if n_jobs == -1:
            cpu = os.cpu_count()
            total_frames = universe.trajectory.n_frames
            chunk_size = total_frames//cpu    
            results = Parallel(n_jobs=-1)(delayed(self.short_fd)(i,  chunk_size = chunk_size, fd = fd, last_cpu = cpu - 1, step = step, verbose = False)
                                          for i in range(cpu))
            #fd.run(step = step) 
            self.features = np.concatenate(results)
            
        elif self.mode == "coordinate":
            #print("Coordinate based features ..")
            #fc = PosArray(universe, selection)
            
            fc = featurecoordinates(atomgroup)
            fc.run(step = step, verbose = verbose)
            self.features = fc.results
            #self.features = fc.flat_data
        if frames is not None:
            if len(self.frames):
                self.features = self.features[self.frames]
        self.input_dim = self.features.shape[1] 
