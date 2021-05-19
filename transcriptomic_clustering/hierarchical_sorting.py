from typing import Dict

import numpy as np
from scipy.cluster.hierarchy import linkage


def hclust(cluster_means: Dict[str, np.ndarray]):
    """
    Performs UPGMA hierarchical clustering

    Parameters
    ----------
    cluster_means:
        mapping of cluster ids to cluster means

    Returns
    -------
    linkage_matrix:
        hierarchical clustering encoded as a linkage matrix
    cluster_names:
        list of cluster names that can be used as labels in dendrogram
    """

    # Parse cluster names
    cluster_names = np.array(list(cluster_means.keys()))

    # Convert dictionary of cluster means to a np.ndarray
    cluster_mean_obs = np.array(list(cluster_means.values()))

    # Run UPGMA hierarchical clustering
    linkage_matrix = linkage(cluster_mean_obs, method = 'average', metric = 'euclidean')

    return (
        linkage_matrix,
        cluster_names
    )