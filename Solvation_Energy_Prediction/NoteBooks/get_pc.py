import deepchem as dc
import pandas as pd
import numpy as np
i = 0
full = pd.read_csv("../DataSets/Full_Free_Solv/full.csv", delimiter=";")
smiles =full['SoluteSMILES'].values[i*10:(i+1)*10]
featurizer = dc.feat.PubChemFingerprint()
features = featurizer.featurize(smiles)
np.save(f'{i}.npy', features)
