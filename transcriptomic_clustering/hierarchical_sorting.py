from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage


def hclust(cluster_means: pd.DataFrame):
    """
    Performs UPGMA hierarchical clustering

    Parameters
    ----------
    cluster_means:
        DataFrame of cluster means indexed by cluster label

    Returns
    -------
    linkage_matrix:
        hierarchical clustering encoded as a linkage matrix
    cluster_names:
        list of cluster names that can be used as labels in dendrogram
    """

    # Run UPGMA hierarchical clustering
    linkage_matrix = linkage(cluster_means, method = 'average', metric = 'euclidean')

    return (
        linkage_matrix,
        cluster_means.index.to_numpy()
    )