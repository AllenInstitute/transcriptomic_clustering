import anndata as ad
import scanpy as sc
import time
from typing import Dict, List, Any, Optional
import transcriptomic_clustering as tc
from transcriptomic_clustering.markers import select_marker_genes
import logging

logger = logging.getLogger(__name__)

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

def pairwise_degs(
        adata_norm: ad.AnnData,
        obs_by_cluster: Dict[Any, List], # a dictionary with keys as cluster names and values as lists of cell names
        thresholds: Dict[str, Any], 
        n_markers: int = 20,
        de_method: str = 'ebayes',
        n_jobs: int = 30,
        chunk_size: Optional[int] = None
        ):
    """
    Perform pairwise differential expression analysis on clusters in an AnnData object.
    Parameters
    ----------
    adata_norm : AnnData
        The normalized AnnData object containing the data.
    cluster_assignments : Dict[Any, List]
        A dictionary where keys are cluster names and values are lists of cell names belonging to those clusters
    chunk_size : Optional[int]
        The size of chunks to process at a time. If None, the entire dataset is processed
    thresholds : Dict[str, Any]
    n_markers : int
        The number of up-regulated and downregulated markers to return for all pairs of clusters
    """
    cluster_by_obs = _cluster_obs_dict_to_list(obs_by_cluster)
    logger.info("Computing Cluster Means")
    tic = time.perf_counter()
    cl_means, present_cl_means, cl_vars = tc.get_cluster_means(adata_norm,
                                                    obs_by_cluster,
                                                    cluster_by_obs,
                                                    chunk_size,
                                                    low_th=thresholds['low_thresh'])
    logger.info(f'Completed Cluster Means')
    toc = time.perf_counter()
    logger.info(f'Cluster Means Elapsed Time: {toc - tic}')

    logger.info('Starting Marker Selection')
    tic = time.perf_counter()
    markers = select_marker_genes(
        cluster_assignments=obs_by_cluster,
        cluster_means=cl_means,
        cluster_variances=cl_vars,
        present_cluster_means=present_cl_means,
        thresholds=thresholds,
        n_markers=n_markers,
        de_method=de_method,
        n_jobs=n_jobs
    )
    logger.info('Completed Marker Selection')
    toc = time.perf_counter()
    logger.info(f'Marker Selection Elapsed Time: {toc - tic}')

    return markers