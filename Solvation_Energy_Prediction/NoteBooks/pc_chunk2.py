#!/usr/bin/env python3
import argparse
import deepchem as dc
import pandas as pd
import numpy as np
import os
from multiprocessing import Pool
from tqdm import tqdm

def process_chunk(args):
    """Process a chunk of SMILES strings and save PubChem fingerprints."""
    i, input_file, output_dir, chunk_size = args
    try:
        full = pd.read_csv(input_file, delimiter=";")
        smiles = full['SoluteSMILES'].values[i*chunk_size:(i+1)*chunk_size]
        
        featurizer = dc.feat.PubChemFingerprint()
        features = featurizer.featurize(smiles)
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'chunk_{i}.npy')
        np.save(output_path, features)
        return (True, i, output_path)
    except Exception as e:
        return (False, i, str(e))

def main():
    parser = argparse.ArgumentParser(description='Process SMILES strings in chunks and generate PubChem fingerprints.')
    parser.add_argument('-f', '--input-file', default='../DataSets/Full_Free_Solv/full.csv',
                       help='Path to input CSV file')
    parser.add_argument('-o', '--output-dir', default='output',
                       help='Output directory for .npy files')
    parser.add_argument('-c', '--chunk-size', type=int, default=10,
                       help='Number of SMILES per chunk')
    parser.add_argument('-p', '--processes', type=int, default=4,
                       help='Number of parallel processes to use')
    
    args = parser.parse_args()
    
    # Calculate number of chunks needed
    full = pd.read_csv(args.input_file, delimiter=";")
    total_smiles = len(full)
    num_chunks = (total_smiles + args.chunk_size - 1) // args.chunk_size
    
    # Prepare arguments for each chunk
    chunk_args = [(i, args.input_file, args.output_dir, args.chunk_size) 
                 for i in range(num_chunks)]
    
    # Process in parallel
    success_count = 0
    with Pool(processes=args.processes) as pool:
        results = list(tqdm(pool.imap(process_chunk, chunk_args), total=num_chunks, desc="Processing chunks"))
    
    # Print summary
    for status, i, msg in results:
        if status:
            print(f"Chunk {i} succeeded: {msg}")
            success_count += 1
        else:
            print(f"Chunk {i} failed: {msg}")
    
    print(f"\nProcessed {success_count}/{num_chunks} chunks successfully")

if __name__ == "__main__":
    main()
