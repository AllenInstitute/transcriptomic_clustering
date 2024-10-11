from typing import Optional, Dict, Set, List, Any, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
from scipy import sparse
import numpy as np
from numpy.random import default_rng
import anndata as ad

import transcriptomic_clustering as tc

import logging
import time
logger = logging.getLogger(__name__)


@dataclass
class FinalMergeKwargs:
    """Dataclass for kwargs in final_merge"""
    pca_kwargs: Dict = field(default_factory = lambda: ({}))
    filter_pcs_kwargs: Dict = field(default_factory= lambda: ({}))
    filter_known_modes_kwargs: Dict = field(default_factory = lambda: ({}))
    project_kwargs: Dict = field(default_factory = lambda: ({}))
    merge_clusters_kwargs: Dict = field(default_factory = lambda: ({}))
    latent_kwargs: Dict = field(default_factory = lambda: ({}))


def sample_clusters(
        adata: ad.AnnData,
        cluster_dict: Dict[Any, List],
        n_samples_per_clust: int = 20,
        random_seed: int = 123
) -> pd.Series:
    """
    Create a cell mask containing up to n random cells per cluster

    Parameters
    ----------
    adata
        AnnData object
    cluster_dict
        Dictionary of cluster assignments. values are lists of cell names 

    Returns
    -------
    pd.Series that is True for cells selected for sampling
    """

    rng = default_rng(random_seed)
    cell_samples_ids = []
    for k, v in cluster_dict.items():
        if len(v) > n_samples_per_clust:
            choices = rng.choice(v, size=(n_samples_per_clust,))
        else:
            choices = v
        cell_samples_ids.extend(choices)
    
    cell_samples = adata.obs_names[cell_samples_ids]

    cell_mask = pd.Series(
        index=adata.obs.index,
        dtype=bool,
    )
    cell_mask[cell_samples] = True
    return cell_mask


def _cluster_obs_dict_to_list(obs_by_cluster: Dict[int, List[int]]) -> List[int]:
    """
    Convert a dictionary of cluster assignments to a list of cluster assignments
    """
    # Find the total number of observations
    max_index = max(max(indices) for indices in obs_by_cluster.values())

    # Initialize the list of clusters with None (or some other default value)
    cluster_by_obs = [None] * (max_index + 1)

    # Fill in the list with the corresponding cluster for each observation
    for cluster, cell_ids in obs_by_cluster.items():
        for obs in cell_ids:
            cluster_by_obs[obs] = cluster

    return cluster_by_obs


def final_merge(
        adata: ad.AnnData,
        cluster_assignments: List,
        marker_genes: Optional[set] = None,
        n_samples_per_clust: int = 20,
        random_seed: Optional[int]=None,
        n_jobs: Optional[int] = 1,
        return_markers_df: Optional[bool] = False,
        final_merge_kwargs: FinalMergeKwargs = FinalMergeKwargs(),
) -> Tuple[List[List[int]], Union[pd.DataFrame, set]]:
    """
    Runs a final merging step on cluster assignment results
    Step1: Using a pre-defined latent space or compute PCA as below:
    * Do PCA on random sample of cells per cluster and selected marker genes
    * Filter PCA results to select top eigenvectors
    * Project to reduced space
    * remove known eigen vector
   Step2: Do differential expression merge

    Parameters
    ----------
    adata
        AnnData object
    cluster_assignments
        List of arrays of cell ids, one array per cluster. This is the result returned by onestep_clust/iter_clust
    
    """

    obs_by_cluster = defaultdict(lambda: [])
    for i, cell_ids in enumerate(cluster_assignments):
        obs_by_cluster[i] = cell_ids
    
    cluster_by_obs = _cluster_obs_dict_to_list(obs_by_cluster)

    if final_merge_kwargs.latent_kwargs.get("latent_component") is None:

        if marker_genes is None:
            raise ValueError("Need marker genes to run PCA")

        cell_mask = sample_clusters(
            adata=adata,
            cluster_dict=obs_by_cluster,
            n_samples_per_clust=n_samples_per_clust,
            random_seed=random_seed
        )
        gene_mask = pd.Series(
            index=adata.var.index,
            dtype=bool
        )
        gene_mask[marker_genes] = True

        # Do PCA on cell samples and marker genes
        logger.info('Computing PCA on cell samples and marker genes')
        tic = time.perf_counter()
        (components, explained_variance_ratio, explained_variance, means) =  tc.pca(
            adata,
            gene_mask=gene_mask,
            cell_select=cell_mask,
            random_state=random_seed,
            **final_merge_kwargs.pca_kwargs
        )
        logger.info(f'Computed {components.shape[1]} principal components')
        toc = time.perf_counter()
        logger.info(f'PCA Elapsed Time: {toc - tic}')

        # Filter PCA
        logger.info('Filtering PCA Components')
        tic = time.perf_counter()
        components = tc.dimension_reduction.filter_components(
            components,
            explained_variance,
            explained_variance_ratio,
            **final_merge_kwargs.filter_pcs_kwargs
        )
        logger.info(f'Filtered to {components.shape[1]} principal components')
        toc = time.perf_counter()
        logger.info(f'Filter PCA Elapsed Time: {toc - tic}')

        # Project
        logger.info("Projecting into PCA space")
        tic = time.perf_counter()
        projected_adata = tc.project(
            adata, components, means,
            **final_merge_kwargs.project_kwargs
        )
        logger.info(f'Projected Adata Dimensions: {projected_adata.shape}')
        toc = time.perf_counter()
        logger.info(f'Projection Elapsed Time: {toc - tic}')

        # Filter Projection
        #Filter Known Modes
        if final_merge_kwargs.filter_known_modes_kwargs:
            logger.info('Filtering Known Modes')
            tic = time.perf_counter()

            projected_adata = tc.filter_known_modes(
                projected_adata,
                **final_merge_kwargs.filter_known_modes_kwargs
            )

            logger.info(f'Projected Adata Dimensions after Filtering Known Modes: {projected_adata.shape}')
            toc = time.perf_counter()
            logger.info(f'Filter Known Modes Elapsed Time: {toc - tic}')
        else:
            logger.info('No known modes, skipping Filter Known Modes')
    
    else:
        logger.info('Extracting latent dims')
        tic = time.perf_counter()   

         ## Extract latent dimensions
        projected_adata = tc.latent_project(adata, **final_merge_kwargs.latent_kwargs)

        toc = time.perf_counter()
        logger.info(f'Extracting latent dims Elapsed Time: {toc - tic}')

    # Merging
    logger.info('Starting Cluster Merging')
    tic = time.perf_counter()
    cluster_assignments_after_merging, markers = tc.merge_clusters(
        adata_norm=adata,
        adata_reduced=projected_adata,
        cluster_assignments=obs_by_cluster,
        cluster_by_obs=cluster_by_obs,
        return_markers_df=return_markers_df,
        n_jobs=n_jobs,
        **final_merge_kwargs.merge_clusters_kwargs
    )
    logger.info(f'Completed Merging')
    toc = time.perf_counter()
    logger.info(f'Merging Elapsed Time: {toc - tic}')

    cluster_assignments_after_merging = list(cluster_assignments_after_merging.values())

    return cluster_assignments_after_merging, markers
