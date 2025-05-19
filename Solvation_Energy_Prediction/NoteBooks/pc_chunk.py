#!/usr/bin/env python3
import argparse
import deepchem as dc
import pandas as pd
import numpy as np
import os

def process_chunk(i, input_file, output_dir, chunk_size=10):
    """Process a chunk of SMILES strings and save PubChem fingerprints.
    
    Args:
        i (int): Chunk index (0-based)
        input_file (str): Path to input CSV file
        output_dir (str): Directory to save output .npy files
        chunk_size (int): Number of SMILES per chunk
    """
    # Read data
    full = pd.read_csv(input_file, delimiter=";")
    smiles = full['SoluteSMILES'].values[i*chunk_size:(i+1)*chunk_size]
    
    # Featurize
    featurizer = dc.feat.PubChemFingerprint()
    features = featurizer.featurize(smiles)
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'chunk_{i}.npy')
    np.save(output_path, features)
    print(f"Saved chunk {i} to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process SMILES strings in chunks and generate PubChem fingerprints.')
    parser.add_argument('-i', '--index', type=int, required=True,
                       help='Chunk index (0-based)')
    parser.add_argument('-f', '--input-file', default='../DataSets/Full_Free_Solv/full.csv',
                       help='Path to input CSV file (default: ../DataSets/Full_Free_Solv/full.csv)')
    parser.add_argument('-o', '--output-dir', default='output',
                       help='Output directory for .npy files (default: output)')
    parser.add_argument('-c', '--chunk-size', type=int, default=10,
                       help='Number of SMILES per chunk (default: 10)')
    
    args = parser.parse_args()
    
    process_chunk(
        i=args.index,
        input_file=args.input_file,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size
    )
