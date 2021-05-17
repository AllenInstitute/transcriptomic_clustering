from typing import Dict

import numpy as np
import pandas as pd
import transcriptomic_clustering as tc
from scipy.cluster.hierarchy import linkage


def hclust(cluster_means: Dict[str, float]):
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

    # Convert dictionary of cluster means to a dataframe of means
    cluster_mean_obs = pd.DataFrame.from_dict(cluster_means, orient='index')

    # Run UPGMA hierarchical clustering
    linkage_matrix = linkage(cluster_mean_obs, method = 'average', metric = 'euclidean')

    # Parse cluster names
    cluster_names = np.array([k for k in cluster_means.keys()])

    return (
        linkage_matrix,
        cluster_names
    )