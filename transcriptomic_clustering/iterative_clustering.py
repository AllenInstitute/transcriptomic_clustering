from typing import Dict, Optional, List, Any
import logging
import os
import tempfile
from dataclasses import dataclass, field

import math
import numpy as np
import scanpy as sc
import anndata as ad
import h5py
import transcriptomic_clustering as tc
from transcriptomic_clustering.onestep_clustering import onestep_clust, OnestepKwargs


logger = logging.getLogger(__name__)


class AnndataIterWriter():
    def __init__(self, filename, initial_chunk, obs, var):
        self.initialize_file(filename, initial_chunk, obs, var)
        self.adata = sc.read_h5ad(filename, backed='r+')

    def initialize_file(self, filename, initial_chunk, obs, var):
        """Uses initial chunk to determine grouptype"""
        with h5py.File(filename, "w") as f:
            ad._io.h5ad.write_attribute(f, "X", initial_chunk)
            ad._io.h5ad.write_attribute(f, "obs", obs)
            ad._io.h5ad.write_attribute(f, "var", var)

    def add_chunk(self, chunk):
        self.adata.X.append(chunk)


def create_filebacked_clusters(adata, clusters, tmp_dir: Optional[str]=None):
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()

    cluster_by_obs = np.zeros((adata.shape[0],))
    for cl_id, idxs in enumerate(clusters):
        cluster_by_obs[idxs] = cl_id

    adata_size_GB = (adata.n_obs * adata.n_vars) * adata.X.dtype.itemsize / (1024 ** 3)
    chunk_size = tc.memory.estimate_chunk_size(
        adata,
        process_memory=adata_size_GB*2,
        percent_allowed=50,
        process_name='create_filebacked_clusters'
    )

    if chunk_size > adata.n_obs:
        writers = {}
        first = [True] * len(clusters)
        for chunk, start, end in adata.chunked_X(chunk_size):
            for cl_id, cell_ids in enumerate(clusters):
                sliced_chunk = chunk[np.where(cluster_by_obs[start:end] == cl_id)]

                if first[cl_id]:
                    filename = f'{adata.filename}_{cl_id}.h5ad'
                    obs = adata[cell_ids, :].obs
                    var = adata[cell_ids, :].var
                    writers[cl_id] = AnndataIterWriter(filename, sliced_chunk, obs, var)
                    first[cl_id] = False
                else:
                    writers[cl_id].add_chunk(sliced_chunk)

        new_adatas = [writers[cl_id].adata for cl_id, _ in enumerate(clusters)]

    else:
        new_adatas = []
        for i, cell_ids in enumerate(clusters):
            filename = f'{adata.filename}_{i}.h5ad'
            logger.debug('Created filebacked AnnData {filename}')
            new_adatas.append(adata[cell_ids, :].copy(filename=filename))

    return new_adatas


def manage_cluster_adata(adata, clusters, tmp_dir: Optional[str]=None):
    """
    Function for managing memory when iterating

    Returns
    -------
    List of new adata objects for each cluster
    """

    # Estimate memory
    if adata.isbacked:
        adata_size_GB = (adata.n_obs * adata.n_vars) * adata.X.dtype.itemsize / (1024 ** 3)
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
    del adata
    if old_filename:
        os.remove(old_filename)
        logger.debug('Deleted filebacked AnnData {old_filename}')

    return new_adatas


def iter_clust(
        norm_adata,
        min_samples: int=4,
        onestep_kwargs: OnestepKwargs=OnestepKwargs(),
        random_seed: Optional[int]=None,
        tmp_dir: Optional[str]=None) -> List[np.ndarray]:
    """
    Function to iteratively call onestep cluster

    Parameters
    ----------
    norm_adata: log normalized adata (see tc.normalization for computation details)
    onestep_kwargs: Dataclass containg keyword arguments for each function (see OnestepKwargs)

    Returns
    -------
    List of arrays of cell ids, one array per cluster

    """
    logger.info('----------Starting Onestep_clust----------')
    clusters = onestep_clust(norm_adata, onestep_kwargs=onestep_kwargs, random_seed=random_seed)
    logger.info('----------Finished Onestep_clust----------')

    # If only one cluster, return. Otherwise iterate.
    if len(clusters) == 1:
        return clusters

    # Generate new cluster_adata objects (slicing Anndata is questionable...)
    logger.info('Managing cluster data')
    cluster_adatas = manage_cluster_adata(norm_adata, clusters)

    # For each existing cluster, generate new clusters from it.
    new_clusters = []
    for cluster_cell_idxs, cluster_adata in zip(clusters, cluster_adatas):
        if len(cluster_cell_idxs) < min_samples:
            new_clusters.append(cluster_cell_idxs)
        else:
            new_subclusters = iter_clust(
                cluster_adata,
                min_samples,
                onestep_kwargs=onestep_kwargs,
                random_seed=random_seed,
                tmp_dir=tmp_dir
            )
            # Map indices of cluster_adata to indices of norm_adata
            cluster_cell_idxs = np.asarray(cluster_cell_idxs)
            new_clusters.extend([cluster_cell_idxs[cell_idxs] for cell_idxs in new_subclusters])

    return new_clusters


def build_cluster_dict(clusters):
    """Builds a cluster dictionary from a list of lists of samples, each represents a cluster."""
    output = {}
    for i in range(len(clusters)):
        output[i + 1] = clusters[i]
    return output
