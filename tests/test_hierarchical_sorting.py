import pytest

import numpy as np
from transcriptomic_clustering.hierarchical_sorting import hclust

@pytest.fixture
def cluster_means():
    """
        test cluster means
    """
    cluster_means = {
        'A': np.array([0.1232, 1.2423, 3.322212, 0.63823]),
        'B': np.array([0.53453, 0.42463, 0.11232, 1.11111]),
        'C': np.array([5.26124, 0.4536, 4.2323, 11.463632]),
        'D': np.array([0.84323, 9.12312, 6.23123, 2.32123]),
        'E': np.array([0.1642, 3.298774, 5.12, 8.58372]),
        'F': np.array([0.00002, 0.11222, 0.8753, 0.75453]),
        'G': np.array([12.290923, 7.39573, 6.3288, 3.421]),
        'H': np.array([3.76237, 8.21321, 0.22221, 0.8792]),
        'I': np.array([6.00005, 0.0042, 14.87999, 4.98]),
        'J': np.array([0.99903, 1.2221, 1.6520, 7.22324]),
        'K': np.array([1.1135, 2.98723, 3.12314, 4.762]),
        'L': np.array([8.01265, 5.2312, 0.2983, 2.87362]),
        'M': np.array([2.5928324, 0.01548, 0.7902, 4.2398]),
    }

    return cluster_means


def test_hclust(cluster_means):

    expected_cluster_names = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'])
    expected_linkage_matrix = np.array([
        [1, 5, 1.045270 , 2],
        [0, 13, 3.035878, 3],
        [9, 10, 3.369086, 2],
        [12, 15, 3.892025, 3],
        [4, 16, 5.380229, 4],
        [7, 11, 5.562442, 2],
        [14, 17, 6.343478, 7],
        [3, 18, 8.498407, 3],
        [19, 20, 9.280318, 10],
        [2, 21,10.474621, 11],
        [6, 22, 12.911395, 12],
        [8, 23, 14.389003, 13],
    ])

    linkage_matrix, cluster_names = hclust(cluster_means)

    np.testing.assert_equal(cluster_names, expected_cluster_names)
    np.testing.assert_almost_equal(linkage_matrix, expected_linkage_matrix, decimal=6)