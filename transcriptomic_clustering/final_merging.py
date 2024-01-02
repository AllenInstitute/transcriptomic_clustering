fromt typing import Optional, Dict, Set, List, Any
from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
from scipy import sparse
import numpy as np
from numpy.random import default_rng
import anndata as ad

import transcriptomic_clustering as tc
from transcriptomic_clustering.iterative_clustering import (
    build_cluster_dict, summarize_final_clusters
)


logger = logging.getLogger(__name__)


@dataclass
class FinalMergeKwargs:
    """Dataclass for kwargs in final_merge"""
    pca_kwargs: Dict = field(default_factory = lambda: ({}))
    filter_pcs_kwargs: Dict = field(default_factory= lambda: ({}))
    filter_known_modes_kwargs: Dict = field(default_factory = lambda: ({}))
    project_kwargs: Dict = field(default_factory = lambda: ({}))
    merge_clusters_kwargs: Dict = field(default_factory = lambda: ({}))

@usage_decorator
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
    Adata 

    Returns
    -------
    pd.Series that is True for cells selected for sampling
    """

    rng = default_rng(random_seed)
    cell_samples = []
    for k, v in cluster_dict.items():
        if len(v) > n_samples_per_clust:
            choices = rng.choice(v, size=(n_samples_per_clust,))
        else:
            choices = v
        cell_samples.extend(choices)

    cell_mask = pd.Series(
        index=adata.obs.index,
        dtype=bool,
    )
    cell_mask[cell_samples] = True
    return cell_mask

@usage_decorator
def final_merge(
        adata: ad.AnnData,
        cluster_assignments: pd.Series,
        marker_genes: Set,
        n_samples_per_clust: int = 20,
        random_seed: Optional[int]=None,
        final_merge_kwargs: FinalMergeKwargs = FinalMergeKwargs(),
) -> pd.DataFrame:
    """
    Runs a final merging step on cluster assignment results
    * Do PCA on random sample of cells per cluster and selected marker genes
    * Filter PCA results to select top eigenvectors
    * Project to reduced space
    * remove known eigen vector
    * Do differential expression merge

    Parameters
    ----------
    
    """

    # Quick data rearranging for convenience
    cluster_dict = defaultdict(lambda: [])
    for cell_name, cl_label in cluster_assignments.iteritems():
        cluster_dict[row].append(cell_name)

    cell_mask = sample_clusters(
        adata=adata,
        cluster_dict=cluster_dict,
        n_samples_per_clust=n_samples_per_clust,
        random_seed=random_seed
    )
    gene_mask = pd.Series(
        index=adata.var.index,
        dtype=bool
    )
    gene_mask[markers] = True

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
    

    # Merging
    logger.info('Starting Cluster Merging')
    tic = time.perf_counter()
    cluster_assignments_after_merging = tc.merge_clusters(
        adata_norm=adata,
        adata_reduced=projected_adata,
        cluster_assignments=cluster_dict,
        cluster_by_obs=cluster_assignments,
        **final_merge_kwargs.merge_clusters_kwargs
    )
    logger.info(f'Completed Merging')
    toc = time.perf_counter()
    logger.info(f'Merging Elapsed Time: {toc - tic}')

    return cluster_assignments_after_merging
