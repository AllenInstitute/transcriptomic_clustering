import sys
import os
import pickle
import pandas as pd
import scanpy as sc
import importlib
import time
import anndata as ad
import numpy as np
import math

# Skip this line if transcriptomic_clustering is installed
sys.path.insert(1, '/allen/programs/celltypes/workgroups/rnaseqanalysis/dyuan/tool/transcriptomic_clustering/')

from transcriptomic_clustering.final_merging import final_merge, FinalMergeKwargs

# Load the data that contains the raw counts in the 'X' slot. If adata.X is normalized, skip the next normalization step.
adata = sc.read('path/to/your/data.h5ad')

# Normalize the data. Skip if adata.X is already normalized.
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.log1p(adata)

# Add scVI latent space to the adata object. Skip if adata.obsm['scVI'] is already present.
scvi = pd.read_csv('path/to/scvi_latent_space.csv', index_col=0)
adata.obsm['scVI'] = np.asarray(scvi)

# loading clustering results
cl_pth = "/path/to/clustering_results"
with open(os.path.join(cl_pth, 'clustering_results.pkl'), 'rb') as f:
    clusters = pickle.load(f)

# marker genes are only needed for computing PCA
with open(os.path.join(cl_pth, 'markers.pkl'), 'rb') as f:
    markers = pickle.load(f)

# The first 4 are for PCA. modify latent_kwargs if using a pre-computed latent space
def setup_merging(): 
    pca_kwargs ={
        # 'cell_select': 30000, # should not use set this for final merging, as we need to sample from each cluster if computing PCA
        'n_comps': 50,
        'svd_solver': 'randomized'
    }
    filter_pcs_kwargs = {
        'known_components': None,
        'similarity_threshold': 0.7,
        'method': 'zscore', 
        'zth': 2,
        'max_pcs': None}
    filter_known_modes_kwargs = {
        'known_modes': None,
        'similarity_threshold': 0.7}
    project_kwargs = {}
    merge_clusters_kwargs = {
        'thresholds': {
            'q1_thresh': 0.5,
            'q2_thresh': None,
            'cluster_size_thresh': 10, 
            'qdiff_thresh': 0.7, 
            'padj_thresh': 0.05, 
            'lfc_thresh': 1, 
            'score_thresh': 100, 
            'low_thresh': 0.6931472, 
            'min_genes': 5
        },
        'k': 4,
        'de_method': 'ebayes'
    }
    latent_kwargs = { # if None: default is running pca, else use the latent_component in adata.obsm
        'latent_component': "scVI"
    }
    
    merge_kwargs = FinalMergeKwargs(
        pca_kwargs = pca_kwargs,
        filter_pcs_kwargs = filter_pcs_kwargs,
        filter_known_modes_kwargs = filter_known_modes_kwargs,
        project_kwargs = project_kwargs,
        merge_clusters_kwargs = merge_clusters_kwargs,
        latent_kwargs = latent_kwargs
    )
    return merge_kwargs

merge_kwargs = setup_merging()

# Run the final merging
clusters_after_merging, markers = final_merge(
    adata, 
    clusters, 
    markers, # required for PCA, but optional if using a pre-computed latent space
    n_samples_per_clust=20, 
    random_seed=2024, 
    final_merge_kwargs=merge_kwargs,
    n_jobs = 30, # modify this to the number of cores you want to use
    return_markers_df = True # return the pair-wise DE results for each cluster pair. If False (default), only return a set of markers (top 20 of up and down regulated genes in each pair comparison)
)