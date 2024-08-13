import scanpy as sc
import pandas as pd
import numpy as np
import transcriptomic_clustering
from transcriptomic_clustering.iterative_clustering import (
    build_cluster_dict, iter_clust, OnestepKwargs
)

# Load the data that contains the raw counts in the 'X' slot. If using normalized data, skip the normalization step.
adata = sc.read('path/to/your/data.h5ad')

# Normalize the data. Skip if adata.X is already normalized.
adata=transcriptomic_clustering.normalize(adata)

# Add scVI latent space to the adata object. Skip if adata.obsm['X_scVI'] is already present.
scvi = pd.read_csv('path/to/scvi_latent_space.csv', index_col=0)
adata.obsm['X_scVI'] = np.asarray(scvi)

# Set up the clustering parameters
def setup_transcriptomic_clustering(): 
    means_vars_kwargs = {
        'low_thresh': 0.6931472,
        'min_cells': 4 
    }
    highly_variable_kwargs = {
        'max_genes': 4000 
    }
    pca_kwargs = { # not used if using a scvi latent space
        'cell_select': 30000, 
        'n_comps': 50,
        'svd_solver': 'randomized'
    }
    filter_pcs_kwargs = { # not used if using a scvi latent space
        'known_components': None,
        'similarity_threshold': 0.7,
        'method': 'zscore', 
        'zth': 2,
        'max_pcs': None,
    }
    filter_known_modes_kwargs = { 
        # 'known_modes': known_modes_df,
        'similarity_threshold': 0.7
    }
    latent_kwargs = {
        'latent_component': "X_scVI"
    }
    cluster_louvain_kwargs = { 
        'k': 15, 
        'nn_measure': 'euclidean',
        'knn_method': 'annoy',
        'louvain_method': 'taynaud', 
        'weighting_method': 'jaccard',
        'n_jobs': 30, 
        'resolution': 1.0, 
    }
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
    onestep_kwargs = OnestepKwargs(
        means_vars_kwargs = means_vars_kwargs,
        highly_variable_kwargs = highly_variable_kwargs,
        pca_kwargs = pca_kwargs,
        filter_pcs_kwargs = filter_pcs_kwargs,
        filter_known_modes_kwargs = filter_known_modes_kwargs,
        latent_kwargs = latent_kwargs,
        cluster_louvain_kwargs = cluster_louvain_kwargs,
        merge_clusters_kwargs = merge_clusters_kwargs
    )
    return onestep_kwargs

onestep_kwargs = setup_transcriptomic_clustering()

# run the iterative clustering. need a tmp folder to store intermediate results
clusters, markers = iter_clust(
    adata,
    min_samples=10, 
    onestep_kwargs=onestep_kwargs,
    random_seed=123,
    tmp_dir="/path/to/your/tmp"
)

cluster_dict = build_cluster_dict(clusters)

adata.obs["scrattch_py_cluster"] = ""
for cluster in cluster_dict.keys():
    adata.obs.scrattch_py_cluster[cluster_dict[cluster]] = cluster

adata.obs.scrattch_py_cluster = adata.obs.scrattch_py_cluster.astype('category')

# save the clustering results
res = pd.DataFrame({'sample_id':adata.obs_names, 'cl': adata.obs.scrattch_py_cluster})
res.to_csv('/path/to/your/clustering_results.csv')
