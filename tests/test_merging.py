import os
import pytest

import numpy as np
import pandas as pd
import anndata as ad
from transcriptomic_clustering import merging
from scipy.sparse import random, csr_matrix
import scanpy as sc

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def tasic_reduced_dim_adata():

    tasic_reduced_path = os.path.join(DATA_DIR, "tasic_reduced_dim.h5ad")
    return sc.read_h5ad(tasic_reduced_path)


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
    obs['cluster_label'] = ['11', 2, 2, '11', '32', '11', 2, '32', 4, '11']
    var_names = [f"var_{i}" for i in range(n_var)]
    var = pd.DataFrame(index=var_names)
    adata = ad.AnnData(csr_matrix(X), obs=obs, var=var)

    return adata


@pytest.fixture
def clusters():
    cluster_assignments = {
        '11': [0, 3, 5, 9],
        2: [1, 2, 6],
        '32': [4, 7],
        4: [8]
    }

    cluster_means = {
        '11': np.array([3., 1.5, 4.]),
        2: np.array([3., 4., 2.]),
        '32': np.array([3., 0.5, 3.]),
        4: np.array([0., 0., 7.]),
    }

    return cluster_means, cluster_assignments

@pytest.fixture
def df_clusters():
    cluster_assignments = {
        '11': [0, 3, 5, 9],
        2: [1, 2, 6],
        '32': [4, 7],
        4: [8]
    }

    cluster_means = pd.DataFrame(
        np.array([[3., 1.5, 4.],
                  [3., 4., 2.],
                  [3., 0.5, 3.],
                  [0., 0., 7.]]),
        index = ['11', 2, '32', 4]
    )

    return cluster_means, cluster_assignments


def test_get_cluster_means(adata, clusters):

    cluster_means, cluster_assignments = clusters

    expected_cluster_means = cluster_means
    obtained_cluster_means = merging.get_cluster_means(adata, cluster_assignments)

    assert set(obtained_cluster_means.keys()) == set(cluster_means.keys())
    for k, v in expected_cluster_means.items():
        assert np.array_equal(obtained_cluster_means[k], expected_cluster_means[k])


def test_merge_two_clusters(clusters):

    cluster_means, cluster_assignments = clusters

    merging.merge_two_clusters(cluster_means,
                               cluster_assignments,
                               label_source=4,
                               label_dest=2)

    expected_cluster_assignments = {
        '11': [0, 3, 5, 9],
        2: [1, 2, 6, 8],
        '32': [4, 7],
    }

    expected_cluster_means = {
        '11': np.array([3., 1.5, 4.]),
        2: (np.array([3., 4., 2.])*3 + np.array([0., 0., 7.])*1)/(3+1),
        '32': np.array([3., 0.5, 3.]),
    }

    assert set(cluster_assignments.keys()) == set(expected_cluster_assignments.keys())
    for k, v in expected_cluster_assignments.items():
        assert np.array_equal(cluster_assignments[k], expected_cluster_assignments[k])

    assert set(cluster_means.keys()) == set(expected_cluster_means.keys())
    for k, v in expected_cluster_means.items():
        assert np.array_equal(cluster_means[k], expected_cluster_means[k])


def test_pdist_normalized():

    cluster_means = {
        '11': [1],
        2: [3],
        '32': [5],
    }
    expected_similarity = np.array(
        [[1, 0.5, 0 ],
         [0.5, 1, 0.5],
         [0,  0.5, 1 ]]
    )

    cluster_means_arr = np.array(list(cluster_means.values()))
    obtained_similarity = merging.pdist_normalized(cluster_means_arr)

    assert np.array_equal(expected_similarity, obtained_similarity)


def test_find_most_similar(df_clusters):

    cluster_means, cluster_assignments = df_clusters
    group_rows = ['32', 4]
    group_cols = ['11', 2, '32', 4]

    similarity_df = merging.calculate_similarity(cluster_means,
                                       group_rows=group_rows,
                                       group_cols=group_cols)

    source_label, dest_label, max_similarity = merging.find_most_similar(similarity_df)

    assert (source_label, dest_label) == ('32', '11')
    assert max_similarity == similarity_df.loc['32']['11']


def test_merge_small_clusters(clusters):

    cluster_means, cluster_assignments = clusters

    expected_cluster_assignments = {
        '11': [0, 3, 5, 9, 4, 7, 8],
        2: [1, 2, 6]
    }

    merging.merge_small_clusters(cluster_means, cluster_assignments, min_size=3)

    assert set(cluster_assignments.keys()) == set(expected_cluster_assignments.keys())
    for k, v in cluster_assignments.items():
        assert np.array_equal(cluster_assignments[k], expected_cluster_assignments[k])


def test_on_tasic_clusters(tasic_reduced_dim_adata):

    adata = tasic_reduced_dim_adata

    cluster_assignments = merging.get_cluster_assignments(
        adata,
        cluster_label_obs="cluster_label_init")
    cluster_means = merging.get_cluster_means(adata, cluster_assignments)

    expected_cluster_assignments = merging.get_cluster_assignments(
        adata,
        cluster_label_obs="cluster_label_after_merging_small")

    expected_cluster_means = merging.get_cluster_means(adata, expected_cluster_assignments)

    merging.merge_small_clusters(cluster_means, cluster_assignments, min_size=6)

    for k, v in cluster_assignments.items():
        assert set(cluster_assignments[k]) == set(expected_cluster_assignments[k])

    assert set(cluster_means.keys()) == set(expected_cluster_means.keys())

    for k, v in expected_cluster_means.items():
        np.allclose(cluster_means[k], expected_cluster_means[k], rtol=1e-6)


def test_calculate_similarity(df_clusters):

    cluster_means, cluster_assignments = df_clusters

    group_rows = ['32', 4]
    group_cols = ['11', 2, '32', 4]
    obtained_similarity = merging.calculate_similarity(cluster_means,
                                       group_rows=group_rows,
                                       group_cols=group_cols)

    expected_similarity = pd.DataFrame(
        [[0.917663, -0.866025, 1.0, 0.5],
         [0.802955, -0.866025, 0.5, 1.0]],
        index=group_rows,columns=group_cols)

    obtained_similarity.equals(expected_similarity)


def test_get_cluster_assignments(adata, clusters):

    cluster_means, cluster_assignments = clusters
    obtained_cluster_assignments = merging.get_cluster_assignments(
        adata,
        cluster_label_obs="cluster_label")

    assert set(cluster_assignments.keys()) == set(obtained_cluster_assignments.keys())
    for k, v in cluster_assignments.items():
        assert set(cluster_assignments[k]) == set(obtained_cluster_assignments[k])
