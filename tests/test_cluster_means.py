import os
import pytest

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
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
        '11': [0, 3, 5, 9],
        '2': [1, 2, 6],
        '32': [4, 7],
        '4': [8]
    }

    cluster_by_obs = np.array(['11', '2', '2', '11', '32', '11', '2', '32', '4', '11'])

    cluster_means = pd.DataFrame(
        np.array([[3., 1.5, 4.],
                  [3., 4., 2.],
                  [3., 0.5, 3.],
                  [0., 0., 7.]]),
        index = ['11', '2', '32', '4']
    )

    present_cluster_means = pd.DataFrame(
        np.array([[0.5, 0.25, 0.75],
                  [(1/3), (2/3), (1/3)],
                  [0.5, 0., .5],
                  [0., 0., 1.]]),
        index = ['11', '2', '32', '4']
    )

    return cluster_means, present_cluster_means, cluster_assignments, cluster_by_obs


def test_get_cluster_means_inmemory(adata, clusters):

    expected_cluster_means, expected_present_cluster_means, cluster_assignments, cluster_by_obs = clusters

    obtained_cluster_means, obtained_present_cluster_means = cm.get_cluster_means(adata, cluster_assignments, cluster_by_obs, low_th=2)

    assert obtained_cluster_means.index.equals(expected_cluster_means.index)
    assert obtained_cluster_means.columns.equals(expected_cluster_means.columns)
    assert np.allclose(obtained_cluster_means.to_numpy(), expected_cluster_means.to_numpy())
    assert obtained_present_cluster_means.index.equals(expected_present_cluster_means.index)
    assert obtained_present_cluster_means.columns.equals(expected_present_cluster_means.columns)
    assert np.allclose(obtained_present_cluster_means.to_numpy(), expected_present_cluster_means.to_numpy())


def test_get_cluster_means_backed(adata, clusters, tmpdir_factory):

    expected_cluster_means, expected_present_cluster_means, cluster_assignments, cluster_by_obs = clusters

    tmpdir = str(tmpdir_factory.mktemp("test_cluster_means"))
    input_file_name = os.path.join(tmpdir, "input.h5ad")

    ad.AnnData(csr_matrix(adata.X)).write(input_file_name) # make tmp input file

    adata = sc.read_h5ad(input_file_name, backed='r')
    obtained_cluster_means, obtained_present_cluster_means = cm.get_cluster_means(adata, cluster_assignments, cluster_by_obs, low_th=2)

    assert obtained_cluster_means.index.equals(expected_cluster_means.index)
    assert obtained_cluster_means.columns.equals(expected_cluster_means.columns)
    assert np.allclose(obtained_cluster_means.to_numpy(), expected_cluster_means.to_numpy())
    assert obtained_present_cluster_means.index.equals(expected_present_cluster_means.index)
    assert obtained_present_cluster_means.columns.equals(expected_present_cluster_means.columns)
    assert np.allclose(obtained_present_cluster_means.to_numpy(), expected_present_cluster_means.to_numpy())