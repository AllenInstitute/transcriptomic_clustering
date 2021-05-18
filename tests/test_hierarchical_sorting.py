import pytest

import numpy as np
from transcriptomic_clustering.hierarchical_sorting import hclust

@pytest.fixture
def cluster_means():
    """
        test cluster means
    """
    cluster_means = {
        'A': 0.1232,
        'B': 0.53453,
        'C': 5.26124,
        'D': 0.84323,
        'E': 0.1642,
        'F': 0.00002,
        'G': 12.290923,
        'H': 3.76237,
        'I': 6.00005,
        'J': 0.99903,
        'K': 1.1135,
        'L': 8.01265,
        'M': 2.5928324,
    }

    return cluster_means


def test_hclust(cluster_means):

    expected_cluster_names = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'])
    expected_linkage_matrix = np.array([
        [0, 4, 0.041, 2],
        [9, 10, 0.11447, 2],
        [5, 13, 0.14368, 3],
        [3, 14, 0.213035, 3],
        [1, 15, 0.4387233, 4],
        [2, 8, 0.73881, 2],
        [16, 17, 0.7797658, 7],
        [7, 12, 1.1695376, 2],
        [11, 18, 2.382005, 3],
        [19, 20, 2.6379283, 9],
        [21, 22, 5.29876750, 12],
        [6, 23, 9.8403522, 13]
    ])

    linkage_matrix, cluster_names = hclust(cluster_means)

    np.testing.assert_equal(cluster_names, expected_cluster_names)
    np.testing.assert_almost_equal(linkage_matrix, expected_linkage_matrix, decimal=6)