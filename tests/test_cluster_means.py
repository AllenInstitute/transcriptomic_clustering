import os
import pytest

import numpy as np
import pandas as pd
import anndata as ad
from transcriptomic_clustering import cluster_means as cm
from scipy.sparse import csr_matrix


@pytest.fixture
def adata():

    X = np.array([
        [0, 1, 3],
        [0, 6, 2],
        [1, 0, 1],
        [4, 4, 7],
        [4, 0, 0],
        [6, 0, 6],
        [8, 6, 3],
        [2, 1, 6],
        [0, 0, 7],
        [2, 1, 0],
    ])
    n_obs = X.shape[0]
    n_var = X.shape[1]
    cell_names = [f"cell_{i}" for i in range(n_obs)]
    obs = pd.DataFrame(index=cell_names)

    var_names = [f"var_{i}" for i in range(n_var)]
    var = pd.DataFrame(index=var_names)
    adata = ad.AnnData(csr_matrix(X), obs=obs, var=var)

    return adata


@pytest.fixture
def clusters():
    cluster_assignments = {
        11: [0, 3, 5, 9],
        2: [1, 2, 6],
        32: [4, 7],
        4: [8]
    }

    cluster_by_obs = np.array([11, 2, 2, 11, 32, 11, 2, 32, 4, 11])

    cluster_means = {
        11: np.array([3., 1.5, 4.]),
        2: np.array([3., 4., 2.]),
        32: np.array([3., 0.5, 3.]),
        4: np.array([0., 0., 7.]),
    }

    present_cluster_means = {
        11: np.array([0.5, 0.25, 0.75]),
        2: np.array([(1/3), (2/3), (1/3)]),
        32: np.array([0.5, 0., .5]),
        4: np.array([0., 0., 1.]),
    }

    return cluster_means, present_cluster_means, cluster_assignments, cluster_by_obs


def test_get_cluster_means_inmemory(adata, clusters):

    cluster_means, present_cluster_means, cluster_assignments, _ = clusters

    expected_cluster_means = cluster_means
    expected_present_cluster_means = present_cluster_means
    obtained_cluster_means, obtained_present_cluster_means = cm.get_cluster_means_inmemory(adata, cluster_assignments, low_th=2)

    assert set(obtained_cluster_means.keys()) == set(cluster_means.keys())
    assert set(expected_present_cluster_means.keys()) == set(present_cluster_means.keys())
    for k, _ in expected_cluster_means.items():
        assert np.array_equal(obtained_cluster_means[k], expected_cluster_means[k])
        assert np.array_equal(obtained_present_cluster_means[k], expected_present_cluster_means[k])

def test_get_cluster_means_chunked(adata, clusters):

    cluster_means, present_cluster_means, cluster_assignments, cluster_by_obs = clusters

    expected_cluster_means = cluster_means
    expected_present_cluster_means = present_cluster_means
    obtained_cluster_means, obtained_present_cluster_means = cm.get_cluster_means_chunked(adata, cluster_assignments, cluster_by_obs, low_th=2, chunk_size=2)

    assert set(obtained_cluster_means.keys()) == set(cluster_means.keys())
    assert set(expected_present_cluster_means.keys()) == set(present_cluster_means.keys())
    for k, _ in expected_cluster_means.items():
        assert np.array_equal(obtained_cluster_means[k], expected_cluster_means[k])
        assert np.array_equal(obtained_present_cluster_means[k], expected_present_cluster_means[k])