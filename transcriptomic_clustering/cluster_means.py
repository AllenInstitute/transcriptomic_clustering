from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import transcriptomic_clustering as tc
import warnings


def get_cluster_means(
        adata: ad.AnnData,
        cluster_assignments: Dict[Any, np.ndarray],
        cluster_by_obs: np.ndarray,
        chunk_size: Optional[int]=None,
        low_th: Optional[int]=1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    if adata.isbacked:
        cluster_means, present_cluster_means = get_cluster_means_backed(adata, cluster_assignments, cluster_by_obs, chunk_size, low_th)
    else:
        if chunk_size:
            warnings.warn("In memory processing does not support chunking. "
                          "Ignoring `chunk_size` argument.")
    
        cluster_means, present_cluster_means = get_cluster_means_inmemory(adata, cluster_assignments, low_th)

    return (cluster_means, present_cluster_means)


def get_cluster_means_inmemory(
        adata: ad.AnnData,
        cluster_assignments: Dict[Any, np.ndarray],
        low_th: Optional[int]=1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute mean gene expression over cells belonging to each cluster in memory
    See description of get_cluster_means() for details
    """

    cluster_means_lst = []
    present_cluster_means_lst = []

    for clust in cluster_assignments.values():
        slice = adata.X[clust, :]
        cluster_means_lst.append(np.asarray(np.mean(slice, axis=0)).ravel())
        present_cluster_means_lst.append(np.asarray(np.mean((slice > low_th), axis=0)).ravel())

    cluster_means = pd.DataFrame(
        np.vstack(cluster_means_lst),
        index=list(cluster_assignments.keys()),
        columns=adata.var.index
    )
    present_cluster_means = pd.DataFrame(
        np.vstack(present_cluster_means_lst),
        index=list(cluster_assignments.keys()),
        columns=adata.var.index
    )

    return cluster_means, present_cluster_means


def get_cluster_means_backed(
        adata: ad.AnnData,
        cluster_assignments: Dict[Any, np.ndarray],
        cluster_by_obs: np.ndarray,
        chunk_size: int,
        low_th: Optional[int]=1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute mean gene expression over cells belonging to each cluster of file-backed data in chunks
    See description of get_cluster_means() for details
    """

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    n_clusters = len(cluster_assignments.keys())

    itemsize = adata.X.dtype.itemsize
    process_memory_estimate = (n_cells * n_genes) * itemsize / (1024 ** 3)
    output_memory_estimate = 2 * (n_clusters * n_genes) * itemsize / (1024 ** 3)
    
    estimated_chunk_size = tc.memory.estimate_chunk_size(
        adata,
        process_memory=process_memory_estimate,
        output_memory=output_memory_estimate,
        percent_allowed=25,
        process_name='get_cluster_means'
    )

    if chunk_size:
        if not(chunk_size >= 1 and isinstance(chunk_size, int)):
            raise ValueError("chunk_size argument must be a positive integer")

        if estimated_chunk_size < chunk_size:
           warnings.warn(f"Selected chunk_size: {chunk_size} is larger than "
                         f"the estimated chunk_size {estimated_chunk_size}. "
                         f"Using chunk_size larger than recommended may result in MemoryError")
    else:
        chunk_size = estimated_chunk_size

    cluster_labels = list([k for k in cluster_assignments.keys()])

    # Get cluster X cell to calculate cluster sums. This allows us to do a chunked calculation of cluster X cell @ cell X gene
    one_hot_cl = get_one_hot_cluster_array(cluster_by_obs, cluster_labels)

    # Get array of cluster sizes for calculating cluster means
    cluster_sizes = np.array([len(cluster_assignments[k]) for k in cluster_labels]).reshape(-1, 1)

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
    cluster_means = pd.DataFrame(cl_means, index=cluster_labels)
    present_cluster_means = pd.DataFrame(present_cl_means, index=cluster_labels)

    return (cluster_means, present_cluster_means)


def get_one_hot_cluster_array(
        cluster_by_obs: np.ndarray,
        cluster_labels: List[Any]
) -> np.ndarray:
    """
    Compute a one-hot array of clusters by cells

    Parameters
    ----------
    cluster_by_obs:
        array of cells with cluster value
    cluster_labels:
        list of cluster labels

    Returns
    -------
    one_hot_cl:
        one-hot array of cells in a cluster (rows=clusters, columns=cells)
    """

    n_clusters = len(cluster_labels)
    cluster_idxs = np.array([cluster_labels.index(cl) for cl in cluster_by_obs])

    b = np.zeros((cluster_by_obs.size, n_clusters))
    b[np.arange(cluster_by_obs.size), cluster_idxs] = 1
    return csr_matrix(b).toarray().T