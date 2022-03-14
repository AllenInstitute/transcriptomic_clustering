from typing import Dict, Optional, List, Any, Union, Sequence
import logging
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass, field
import time

import math
import numpy as np
import scanpy as sc
import scipy as scp
import anndata as ad
import h5py
import transcriptomic_clustering as tc
from transcriptomic_clustering.onestep_clustering import onestep_clust, OnestepKwargs
from transcriptomic_clustering.iter_writer import AnnDataIterWriter


logger = logging.getLogger(__name__)


def create_filebacked_clusters(adata, clusters, tmp_dir: Path):
    """
    Handles creating a new AnnData filebacked object for each cluster

    Parameters
    ----------
    adata: adata object
    clusters: list of lists of cell ids in each cluster
    tmp_dir: directory to write new cluster adata objects
             if too large for memory

    Returns
    -------
    List of new filebacked adata objects for each cluster

    """

    logger.debug(f"creating tmp files in {tmp_dir}")

    cluster_by_obs = np.zeros((adata.shape[0],))
    for cl_id, idxs in enumerate(clusters):
        cluster_by_obs[idxs] = cl_id

    if not adata.is_view:  # .X on view will try to load entire X into memory
        itemsize = adata.X.dtype.itemsize
    else:
        itemsize = np.dtype(np.float64).itemsize
    adata_size_GB = (adata.n_obs * adata.n_vars) * itemsize / (1024 ** 3)
    chunk_size = tc.memory.estimate_chunk_size(
        adata,
        process_memory=adata_size_GB*2,
        percent_allowed=50,
        process_name='create_filebacked_clusters'
    )

    old_filename = Path(adata.filename).stem
    # TODO: Hack
    chunk_size = 10000
    # if chunk_size < adata.n_obs:
    writers = {}
    first = [True] * len(clusters)
    for chunk, start, end in adata.chunked_X(chunk_size):
        for cl_id, cell_ids in enumerate(clusters):
            sliced_chunk = chunk[np.where(cluster_by_obs[start:end] == cl_id)]
            if sliced_chunk.shape[0] > 0:
                if first[cl_id]:
                    filename = Path(tmp_dir) / f'{old_filename}_{cl_id}.h5ad'
                    obs = adata.obs.iloc[cell_ids]
                    var = adata.var
                    writers[cl_id] = AnnDataIterWriter(filename, sliced_chunk, obs, var)
                    first[cl_id] = False
                else:
                    writers[cl_id].add_chunk(sliced_chunk)

    new_adatas = [writers[cl_id].adata for cl_id, _ in enumerate(clusters)]

    # else:
    #     new_adatas = []
    #     for i, cell_ids in enumerate(clusters):
    #         filename = Path(tmp_dir) / f'{old_filename}_{i}.h5ad'
    #         logger.debug(f'Created filebacked AnnData {filename}')
    #         new_adatas.append(adata[cell_ids, :].copy(filename=filename))

    return new_adatas


def manage_cluster_adata(adata, clusters, tmp_dir: Path):
    """
    Function for managing memory when iterating
    Will decide whether to load cluster into memory or write new AnnData file

    Parameters
    ----------
    adata: adata object
    clusters: list of lists of cell ids in each cluster
    tmp_dir: directory to write new cluster adata objects
             if too large for memory

    Returns
    -------
    List of new adata objects for each cluster
    """

    # Estimate memory
    if adata.isbacked:
        if not adata.is_view:  # .X on view will try to load entire X into memory
            itemsize = adata.X.dtype.itemsize
        else:
            itemsize = np.dtype(np.float64).itemsize
        adata_size_GB = (adata.n_obs * adata.n_vars) * itemsize / (1024 ** 3)
        necessary_memory_estimate_GB =  adata_size_GB * 6 # arbritrary factor of 6 to handle whole pipeline in memory
        memory_available_GB = tc.memory.get_available_memory_GB()
        filebacked_clusters = (memory_available_GB < necessary_memory_estimate_GB)
    else:
        filebacked_clusters = False

    if filebacked_clusters:
        logger.info('Creating Filebacked Cluster AnnDatas')
        new_adatas = create_filebacked_clusters(adata, clusters, tmp_dir=tmp_dir)
    else:
        logger.info('Creating In Memory Cluster AnnDatas')
        if adata.isbacked:
            new_adatas = [adata[cell_ids,:].to_memory() for cell_ids in clusters]
        else:
            new_adatas = [adata[cell_ids,:].copy() for cell_ids in clusters]
    
    # remove old adata
    old_filename = adata.filename
    # TODO: Hack
    if str(old_filename)[-15:] != "normalized.h5ad":
        del adata
        if old_filename:
            os.remove(old_filename)
            logger.debug(f'Deleted filebacked AnnData {old_filename}')

    return new_adatas


def iter_clust(
        norm_adata,
        min_samples: int=4,
        onestep_kwargs: OnestepKwargs=OnestepKwargs(),
        random_seed: Optional[int]=None,
        tmp_dir: Optional[str]=None) -> List[np.ndarray]:
    """
    Function to call onestep_clustering, and recursively call iterclust on each generated cluster
    if the cluster has n cells > min_samples

    Parameters
    ----------
    norm_adata: log normalized adata (see tc.normalization for computation details)
    min_samples: minimum number of obs (cells) to call onestep clust on
    onestep_kwargs: Dataclass containg keyword arguments for each function (see OnestepKwargs)
    random_seed: random seed for repeatability
    tmp_dir: directory to write temporary subcluster files (necessary for large datasets)

    Returns
    -------
    List of arrays of cell ids, one array per cluster

    """
    tic = time.perf_counter()
    logger.info('----------Starting Onestep_clust----------')
    clusters, markers = onestep_clust(norm_adata, onestep_kwargs=onestep_kwargs, random_seed=random_seed)
    logger.info('----------Finished Onestep_clust----------')
    toc = time.perf_counter()
    logger.info(f'Onestep Clustering Elapsed Time: {toc - tic}')
    logger.info(
        f'Split cluster of size {norm_adata.n_obs} into new clusters with sizes:\n{[len(cluster) for cluster in clusters]}'
    )

    # If only one cluster, return. Otherwise iterate.
    if len(clusters) == 1:
        return clusters, markers

    # Generate new cluster_adata objects (slicing Anndata is questionable...)
    logger.info('----Managing cluster data----')
    tic = time.perf_counter()
    cluster_adatas = manage_cluster_adata(norm_adata, clusters, tmp_dir=tmp_dir)
    toc = time.perf_counter()
    logger.info(f'Managing Cluster Data Elapsed Time: {toc - tic}')

    # For each existing cluster, generate new clusters from it.
    new_clusters = []
    for cluster_cell_idxs, cluster_adata in zip(clusters, cluster_adatas):
        if len(cluster_cell_idxs) < min_samples:
            new_clusters.append(cluster_cell_idxs)
        else:
            new_subclusters, new_markers = iter_clust(
                cluster_adata,
                min_samples,
                onestep_kwargs=onestep_kwargs,
                random_seed=random_seed,
                tmp_dir=tmp_dir
            )
            # Map indices of cluster_adata to indices of norm_adata
            cluster_cell_idxs = np.asarray(cluster_cell_idxs)
            new_clusters.extend([cluster_cell_idxs[cell_idxs] for cell_idxs in new_subclusters])
            # Combine markers
            markers.update(new_markers)

    return new_clusters, markers


def build_cluster_dict(clusters: List[Union[np.ndarray, Sequence]]) -> Dict[int, List]:
    """
    Builds a dictionary from a list of lists of samples in each cluster

    Parameters
    ----------
    clusters: list of lists of samples, one list per cluster

    Returns
    -------
    Dict[int, List of samples]

    """
    output = {}
    for i in range(len(clusters)):
        output[i + 1] = np.asarray(clusters[i]).tolist()
    return output


def summarize_final_clusters(
        norm_adata: ad.AnnData,
        clusters: List[np.ndarray],
        de_thresholds: Dict[str, Any],
        low_th: float=1,
        de_method: str='ebayes'):
    """
    Helper function to return differential expression and linkage/labels for final clusters

    Parameters
    ----------
    norm_adata: normalized adata
    clusters: list of lists of samples, one list per cluster (output of iter_clust or onestep_clust)
    de_thresholds: thresholds for differential expression
    low_th: minimum gene expression value for computing means and variances
        (onestep_kwargs.merge_clusters_kwargs['thresholds']['low_thresh'] or
         onestep_kwargs.means_vars_kwargs['low_th'])
    de_method: differential expression method

    Returns
    -------
    Pandas dataframe indexed by (cluster A, cluster B) containing:
        total score, upscore, downscore, upgenes, downgenes, upnum, downnum
    linkage and labels from scipy.cluster.dendogram

    """

    # Build cluster dict and cluster_by_obs
    cluster_dict = build_cluster_dict(clusters)

    cluster_by_obs = np.zeros(norm_adata.n_obs, dtype=int)
    for cluster, obs in cluster_dict.items():
        cluster_by_obs[obs] = cluster

    # get cluster means and variances and sizes
    cl_means, present_cl_means, cl_vars = tc.get_cluster_means(
        norm_adata,
        cluster_dict,
        cluster_by_obs,
        low_th=low_th
    )
    cl_size = {k: len(v) for k, v in cluster_dict.items()}

    # all pairs [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3), etc]
    cluster_ids = list(cluster_dict.keys())
    pairs = [(a, b) for idx, a in enumerate(cluster_ids) for b in cluster_ids[idx + 1:]]

    # Compute differential expression scores
    de_thresholds = de_thresholds.copy()
    de_thresholds.pop('score_thresh', None)
    de_thresholds.pop('low_thresh', None)

    # Compute differential expression for all pairs
    if de_method == 'ebayes':
        de_table = tc.de_pairs_ebayes(
            pairs,
            cl_means,
            cl_vars,
            present_cl_means,
            cl_size,
            de_thresholds,
        )
    elif de_method == 'chisq':
        de_table = tc.de_pairs_chisq(
            pairs,
            cl_means,
            present_cl_means,
            cl_size,
            de_thresholds,
        )
    else:
        raise ValueError(f'Unknown de_method {de_method}, must be one of [chisq, ebayes]')

    # Get linkage
    linkage, labels = tc.hclust(cl_means)

    return de_table, linkage, labels