from typing import Any, Dict, Optional, Tuple

import anndata as ad
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
import transcriptomic_clustering as tc


def get_cluster_means(
        adata: ad.AnnData,
        cluster_assignments: Dict[Any, np.ndarray],
        cluster_by_obs: np.ndarray,
        chunk_size: Optional[int]=None,
        low_th: Optional[int]=1
) -> Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
    """
    Compute mean gene expression over cells belonging to each cluster

    Parameters
    ----------
    adata:
        AnnData with X matrix and annotations
    cluster_assignments:
        map of cluster label to cell ids
    cluster_by_obs:
        array of cells with cluster value
    chunk_size:
        number of observations to process in a single chunk
    low_th:
        minimum expression value used to filter for expressed genes

    Returns
    -------
    cluster_means:
        map of cluster label to mean expressions (array of size n_genes)
    present_cluster_means:
        map of cluster label to mean of expressions present filtered by low_th (array of size n_genes)
    """

    # Estimate memory
    if not chunk_size:
        n_obs = adata.n_obs
        n_vars = adata.n_vars
        n_clusters = len(cluster_assignments.keys())
        itemsize = adata.X.dtype.itemsize
        process_memory_estimate = (n_obs * n_vars) * itemsize / (1024 ** 3)
        output_memory_estimate = 2 * (n_clusters * n_vars) * itemsize / (1024 ** 3)
        
        chunk_size = tc.memory.estimate_chunk_size(
            adata,
            process_memory=process_memory_estimate,
            output_memory=output_memory_estimate,
            percent_allowed=25,
            process_name='get_cluster_means'
        )

    if chunk_size >= adata.n_obs:
        cluster_means, present_cluster_means = get_cluster_means_inmemory(adata, cluster_assignments, low_th)
    else:
        cluster_means, present_cluster_means = get_cluster_means_chunked(adata, cluster_assignments, cluster_by_obs, chunk_size, low_th)

    return (cluster_means, present_cluster_means)


def get_cluster_means_inmemory(
        adata: ad.AnnData,
        cluster_assignments: Dict[Any, np.ndarray],
        low_th: Optional[int]=1
) -> Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
    """
    Compute mean gene expression over cells belonging to each cluster in memory
    See description of get_cluster_means() for details
    """

    cluster_means = {}
    present_cluster_means = {}

    for label, idxs in cluster_assignments.items():
        adata_view = adata[idxs, :]
        X = adata_view.X
        cluster_means[label] = np.asarray(np.mean(X, axis=0)).ravel()
        present_cluster_means[label] = np.asarray(np.mean((X > low_th), axis=0)).ravel()

    return (cluster_means, present_cluster_means)



def get_cluster_means_chunked(
        adata: ad.AnnData,
        cluster_assignments: Dict[Any, np.ndarray],
        cluster_by_obs: np.ndarray,
        chunk_size: int,
        low_th: Optional[int]=1
) -> Tuple[Dict[Any, np.ndarray], Dict[Any, np.ndarray]]:
    """
    Compute mean gene expression over cells belonging to each cluster in chunks
    See description of get_cluster_means() for details
    """

    sorted_cluster_labels = sorted(cluster_assignments.keys())
    n_clusters = len(cluster_assignments.keys())
    n_genes = adata.shape[1]

    # Get cluster X cell to calculate cluster sums. This allows us to do a chunked calculation of cluster X cell @ cell X gene
    one_hot_cl = get_one_hot_cluster_array(cluster_by_obs, sorted_cluster_labels)

    # Get array of cluster sizes for calculating cluster means
    cluster_sizes = np.array([len(cluster_assignments[k]) for k in sorted_cluster_labels]).reshape(-1, 1)

    # Calculate cluster and present cluster sums 
    cluster_sums = np.zeros((n_clusters, n_genes))
    present_cluster_sums = np.zeros((n_clusters, n_genes))
    for chunk, start, end in adata.chunked_X(chunk_size):
        cluster_sums += one_hot_cl[:, start:end] @ chunk
        present_cluster_sums += one_hot_cl[:, start:end] @ (chunk > low_th)

    # Calculate means
    cl_means = cluster_sums / cluster_sizes
    present_cl_means = present_cluster_sums / cluster_sizes

    # Convert to desired output
    cluster_means = {}
    present_cluster_means = {}

    for i in range(n_clusters):
        k = sorted_cluster_labels[i]
        cluster_means[k] = cl_means[i]
        present_cluster_means[k] = present_cl_means[i]

    return (cluster_means, present_cluster_means)



def get_one_hot_cluster_array(
        cluster_by_obs: np.ndarray,
        sorted_cluster_labels: np.ndarray
) -> np.ndarray:
    """
    Compute a one-hot sparse array of clusters by cells

    Parameters
    ----------
    cluster_by_obs:
        array of cells with cluster value
    sorted_cluster_labels:
        sorted list of cluster labels

    Returns
    -------
    one_hot_cl:
        one-hot array of cells in a cluster (rows=clusters, columns=cells)
    """

    n_clusters = len(sorted_cluster_labels)
    cluster_idxs = np.array([sorted_cluster_labels.index(cl) for cl in cluster_by_obs])

    b = np.zeros((cluster_by_obs.size, n_clusters))
    b[np.arange(cluster_by_obs.size), cluster_idxs] = 1
    return csr_matrix(b).toarray().T