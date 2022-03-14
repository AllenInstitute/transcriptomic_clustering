import tempfile
from pathlib import Path
import os
import shutil
import json
import logging

import matplotlib.pyplot as plt

import numpy as np
import scipy as scp
import scanpy as sc
import pandas as pd
import transcriptomic_clustering as tc
from transcriptomic_clustering.iterative_clustering import (
    build_cluster_dict, iter_clust, OnestepKwargs, summarize_final_clusters
)


logger = logging.getLogger(__name__)


def setup_filelogger(logfile: Path):
    fhandler = logging.FileHandler(filename=logfile, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fhandler.setFormatter(formatter)
    fhandler.setLevel(logging.DEBUG)

    root_logger = logging.getLogger()
    root_logger.addHandler(fhandler)


def run_iter_clust():
    """
    
    """
    # Paths
    tc_data_path = Path('/localssd/marmot/matt_dev/tc_data')
    rm_eigen_path = tc_data_path / "macosko" / "rm.eigen.csv"
    normalized_adata_path = tc_data_path / "macosko" / 'Macosko_BICCN.fbm.1004.fixed.selgene_normalized.h5ad'
    clusters_path = tc_data_path / "macosko" / "Macosko_BICCN.fbm.1004.fixed.selgene_clusters.csv"
    markers_path = tc_data_path / "macosko" / "Macosko_BICCN.fbm.1004.fixed.selgene_markers.csv"
    log_file_path = tc_data_path / "macosko" / "clustering.log"
    top_tmp_dir = tc_data_path / 'tmp_data' / 'MacoskoTmp'

    # Setup logging
    setup_filelogger(log_file_path)

    # Setup tmpfile
    tmp_dir = tempfile.mkdtemp(dir=top_tmp_dir)
    logger.debug(f"tmp_dir: {str(tmp_dir)}")

    # Set memory params
    tc.memory.set_memory_limit(GB=100)
    tc.memory.allow_chunking = True

    # Log info
    logger.debug('debug test')
    logger.info(f"normalized_adata: {normalized_adata_path}")
    logger.info(f"rm eigen: {rm_eigen_path}")
    logger.info(f"temporary directory: {tmp_dir}")

    # Load normalized adata
    normalized_adata = sc.read(normalized_adata_path, backed='r')
    normalized_adata.var_names_make_unique()

    # Load rm.eigien
    match = lambda a, b: [ b.index(x) if x in b else None for x in a ]
    rm_eigen = pd.read_csv(rm_eigen_path)
    rm_eigen_df = rm_eigen.set_index("Unnamed: 0").reindex(normalized_adata.obs.index)
    logger.info(f'rm_eigen mapped: {rm_eigen_df[0:5]}')

    # Assign kwargs. Any unassigned args will be set to their respective function defaults
    means_vars_kwargs = {
        'low_thresh': 1,
        'min_cells': 4
    }
    highly_variable_kwargs = {
        'max_genes': 3000
    }

    pca_kwargs = {
        'cell_select': 500000,
        'n_comps': 200,
        'svd_solver': 'randomized'
    }

    filter_pcs_kwargs = {
        'known_components': None,
        'similarity_threshold': 0.7,
        'method': 'elbow', #'elbow' or 'zscore'
        'zth': 1.3,
        'max_pcs': 20,
    }

    # project_kwargs = {
        
    # }

    # Leave empty if you don't want to use known_modes
    filter_known_modes_kwargs = {
        'known_modes': rm_eigen_df,
        'similarity_threshold': 0.7
    }

    cluster_louvain_kwargs = {
        'k': 15, # number of nn
        'nn_measure': 'euclidean',
        'knn_method': 'annoy',
        'louvain_method': 'vtraag',
        'weighting_method': 'jaccard',
        'n_jobs': 8, # cpus
        'resolution': 1., # resolution of louvain for taynaud method
    }

    merge_clusters_kwargs = {
        'thresholds': {
            'q1_thresh': 0.4,
            'q2_thresh': 0.7,
            'cluster_size_thresh': 20,
            'qdiff_thresh': 0.7,
            'padj_thresh': 0.05,
            'lfc_thresh': 1.0,
            'score_thresh': 150,
            'low_thresh': 1,
            'min_genes': 5,
        },
        'k': 2, # number of nn for de merge 
        'de_method': 'ebayes',
        'n_markers': 20,
    }

    onestep_kwargs = OnestepKwargs(
        means_vars_kwargs = means_vars_kwargs,
        highly_variable_kwargs = highly_variable_kwargs,
        pca_kwargs = pca_kwargs,
        filter_pcs_kwargs = filter_pcs_kwargs,
    #     project_kwargs = project_kwargs,
        filter_known_modes_kwargs = filter_known_modes_kwargs,
        cluster_louvain_kwargs = cluster_louvain_kwargs,
        merge_clusters_kwargs = merge_clusters_kwargs
    )

    # Run Iter Clust
    clusters, markers = iter_clust(
        normalized_adata,
        min_samples=4,
        onestep_kwargs=onestep_kwargs,
        random_seed=345,
        tmp_dir=tmp_dir
    )

    # Log cluster size
    logger.info(f'final number of clusters: {len(clusters)}')
    cl_sizes = [len(cluster) for cluster in clusters]
    logger.info(f'final cluster sizes: {cl_sizes}')
    logger.info(f'max cluster size: {np.max(cl_sizes)}')

    # Save clusters to csv
    cluster_dict = build_cluster_dict(clusters)
    cluster_by_obs = np.zeros(normalized_adata.n_obs, dtype=int)
    for cluster, obs in cluster_dict.items():
        cluster_by_obs[obs] = cluster

    df = pd.DataFrame(data=cluster_by_obs, index = [normalized_adata.obs.index], columns=["cl"])
    df.to_csv(clusters_path)

    # Save Markers to csv
    df_m = pd.Series(data=list(markers), name="markers")
    df_m.to_csv(markers_path, header=True)

    logger.info(f"Don't forget to delete temporary directory {tmp_dir}")


if __name__ == "__main__":
    run_iter_clust()