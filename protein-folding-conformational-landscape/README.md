# Mapping Conformational Landscape in Protein Folding

This repository contains code and notebooks used in the study:  
[**"Mapping conformational landscape in protein folding: Benchmarking dimensionality reduction and clustering techniques on the Trp-Cage mini-protein"**](https://doi.org/10.1016/j.bpc.2025.107389)

## Overview

The primary goal of this project is to compare and benchmark widely used dimensionality reduction and clustering techniques for analyzing the conformational dynamics of the Trp-Cage mini-protein.

We used D.E. Shaw, Trp-Cage 208 micro-sec simulation data to:
- Apply and compare dimensionality reduction techniques like PCA, TICA, and VAE.
- Cluster conformations using HDBSCAN, K-means, Hierarchical clustering, and Gaussian Mixture Models.


<pre>
protein-folding-conformational-landscape/
├── README.md                          # Documentation and usage guide
├── Projections_Comparison.ipynb       # Dimensionality reduction with PCA, TICA, and VAE
├── Clustering.ipynb                   # Clustering using HDBSCAN, K-means, Hierarchical, GMM
└── Scripts/                           # Supporting Python scripts and utility functions
</pre>


