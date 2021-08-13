from typing import Dict, Optional, List, Any
import logging
import tempfile
from dataclasses import dataclass

import numpy as np
import scanpy as sc
import anndata as ad
import transcriptomic_clustering as tc
import h5py


logger = logging.getLogger(__name__)


class AnndataIterWriter():
    def __init__(self, filename, initial_chunk, obs, var):
        self.adata = self.initialize_file(filename, initial_chunk, obs, var)

    def initialize_file(self, filename, initial_chunk, obs, var):
        """Uses initial chunk to determine grouptype"""
        with h5py.File(filename, "w") as f:
            ad._io.h5ad.write_attribute(f, "X", initial_chunk)
            ad._io.h5ad.write_attribute(f, "obs", obs)
            ad._io.h5ad.write_attribute(f, "var", var)

        self.adata = sc.read_h5ad(filename, backed='r+')

    def add_chunk(self, chunk):
        self.adata.X.append(chunk)


@dataclass
class OnestepKwargs:
    """Dataclass for kwargs in onestep_clust"""
    means_vars_kwargs: Dict = {}
    highly_variable_kwargs: Dict = {}
    pca_kwargs: Dict = {}
    filter_known_modes_kwargs: Dict = {}
    project_kwargs: Dict = {}
    cluster_louvain_kwargs: Dict = {}
    merge_clusters_kwargs: Dict = {}


def onestep_clust(
        norm_adata: sc.AnnData,
        onestep_kwargs: OnestepKwargs=OnestepKwargs(),
        random_seed: Optional[int]=None) -> List[np.ndarray]:
    """
    Performs an entire clustering step
    * get mean and variance of each gene
    * determine highly variable genes
    * do pca on a sample of cells
    * filter known pca components
    * cluster cells using louvain clustering
    * merge clusters

    Parameters
    ----------
    norm_adata: log normalized adata (see tc.normalization for computation details)
    onestep_kwargs: Dataclass containg keyword arguements for each function (see OnestepKwargs)
    random_seed: random_seed for functions that use a random seed/random state

    Returns
    -------
    List of arrays of cell ids, one array per cluster

    """
    logger.info('Starting onestep clustering')

    # Highly Variable
    logger.info('Computing Means and Variances of genes')
    means, variances, gene_mask = tc.get_means_vars_genes(
        adata=norm_adata,
        **onestep_kwargs.means_vars_kwargs
    )
    logger.info('Computing Highly Variable Genes')
    highly_variable_mask = tc.highly_variable_genes(
        adata=norm_adata,
        means=means,
        variances=variances,
        gene_mask=gene_mask,
        **onestep_kwargs.highly_variable_kwargs
    )

    logger.info('Computing PCA')
    #PCA
    (components, explained_variance_ratio, explained_variance, means) =  tc.pca(
        norm_adata,
        gene_mask=highly_variable_mask,
        random_state=random_seed,
        **onestep_kwargs.pca_kwargs
    )
    logger.info(f'Computed {components.shape[1]} principal components')
    
    #Filter Known Modes
    if onestep_kwargs.filter_known_modes_kwargs is not None:
        logger.info('Filtering Known Modes')
        components = tc.filter_known_modes(components, **onestep_kwargs.filter_known_modes_kwargs)
    else:
        logger.info('No known modes, skipping Filter Known Modes')
    
    #Projection
    logger.info("Projecting normalized adata into PCA space")
    projected_adata = tc.project(
        norm_adata,
        components, means,
        **onestep_kwargs.project_kwargs
    )
    logger.info(f'Projected Adata Dimensions: {projected_adata.shape}')

    #Louvain Clustering
    logger.info('Starting Louvain Clustering')
    cluster_by_obs, obs_by_cluster, graph, qc = tc.cluster_louvain(
        projected_adata,
        random_seed=random_seed,
        **onestep_kwargs.project_kwargs
    )
    logger.info(f'Completed Louvain Clustering, found {len(obs_by_cluster.keys)} clusters')

    #Merging
    cluster_sizes_before_merging = {k: len(v) for k, v in obs_by_cluster.items()}
    logger.info('Starting Cluster Merging')
    cluster_assignments_after_merging = tc.merge_clusters(
        adata_norm=norm_adata,
        adata_reduced=projected_adata,
        cluster_assignments=obs_by_cluster,
        cluster_by_obs=cluster_by_obs,
        **onestep_kwargs.merge_clusters_kwargs
    )
    
    logger.info('Completed Cluster Merging')
    logger.info('Completed One Step Clustering')
    return list(cluster_assignments_after_merging.values())


def create_filebacked_clusters(adata, clusters, tmp_dir: Optional[str]=None):
    if tmp_dir is None:
        tmp_dir = tempfile.mkdtemp()
    
    cluster_by_obs = np.zeros((adata.shape[0],))
    for cl_id, idxs in enumerate(clusters):
        cluster_by_obs[idxs] = cl_id
    
    adata_size_GB = (adata.n_obs * adata.n_vars) * adata.itemsize / (1024 ** 3)
    chunk_size = tc.memory.estimate_chunk_size(
        adata,
        process_memory=adata_size_GB*2,
        percent_allowed=50,
        process_name='create_filebacked_clusters'
    )

    if chunk_size > 1:
        writers = {}
        first = True
        for chunk, start, end in adata.chunked_X(chunk_size):
            for cl_id, cell_ids in enumerate(clusters):
                sliced_chunk = chunk[np.where(cluster_by_obs[start, end] == cl_id)]

                if first:
                    filename = f'{adata.filename}_{cl_id}.h5ad'
                    obs = adata[cell_ids, :].obs
                    var = adata[cell_ids, :].var
                    writers[cl_id] = AnndataIterWriter(filename, sliced_chunk, obs, var)
                else:
                    writers[cl_id].add_chunk(sliced_chunk)
                    
        new_adatas = [writers[cl_id].adata for cl_id, _ in enumerate(clusters)]

    else:
        new_adatas = []
        for i, cell_ids in enumerate(clusters):
            filename = f'{adata.filename}_{i}.h5ad'
            new_adatas.append(adata[cell_ids, :].copy(filename=filename))
    
    return new_adatas
    

def manage_cluster_adata(norm_adata, clusters, tmp_dir: Optional[str]=None):
    """
    Function for managing memory when iterating

    Returns
    -------
    List of new adata objects for each cluster
    """

    # Estimate memory
    if norm_adata.isbacked():
        norm_adata_size_GB = (norm_adata.n_obs * norm_adata.n_vars) * norm_adata.itemsize / (1024 ** 3)
        necessary_memory_estimate_GB =  norm_adata_size_GB * 6 # arbritrary factor of 6 to handle whole pipeline in memory
        memory_available_GB = tc.memory.get_available_memory_GB()
        filebacked_clusters = (memory_available_GB < necessary_memory_estimate_GB)
    else:
        filebacked_clusters = False

    if filebacked_clusters:
        new_adatas = create_filebacked_clusters(norm_adata, clusters, tmp_dir=tmp_dir)
    else:
        new_adatas = [norm_adata[cell_ids,:] for cell_ids in clusters]

    return new_adatas
        


def iter_cluster(
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
    clusters = onestep_clust(norm_adata, onestep_kwargs=onestep_kwargs, random_seed=random_seed)

    # If only one cluster, return. Otherwise iterate.
    if len(clusters) == 1:
        return clusters

    # Generate new cluster_adata objects (slicing Anndata is questionable...)
    cluster_adata = manage_cluster_adata(norm_adata, clusters)
    del norm_adata
    
    # For each existing cluster, generate new clusters from it.
    new_clusters = []
    for cluster_cell_id_array, cluster_adata in zip(clusters, cluster_adata):
        if len(cluster_cell_id_array) < min_samples:
            new_clusters.append(cluster_cell_id_array)
        else:
            new_clusters.extend(
                iter_cluster(
                    cluster_adata,
                    min_samples,
                    onestep_kwargs=onestep_kwargs,
                    random_seed=random_seed,
                    tmp_dir=tmp_dir
                )
            )
    
    return new_clusters


def build_cluster_dict(clusters):
    """Builds a cluster dictionary from a list of lists of samples, each represents a cluster."""
    output = {}
    for i in range(len(clusters)):
        output[i + 1] = clusters[i].tolist()
    return output
