import scanpy as sc
import numpy as np
import pytest
import csv
import os

from mock import patch
from unittest.mock import MagicMock

import transcriptomic_clustering as tc
from transcriptomic_clustering.clustering import cluster_louvain_phenograph

DIR_NAME = os.path.dirname(__file__)
TEST_RANDOM_SEED = 5
TEST_K = 15

@pytest.fixture
def pca_data():
    return sc.read_csv(os.path.join(DIR_NAME, "data/output_dimreduction_pca.csv"), first_column_names=True)

@pytest.fixture
def sample_graph():
    return sc.read_h5ad(os.path.join(DIR_NAME, "data/sample_graph.h5ad"))

@pytest.fixture
def annoy_taynaud_sample_partition():
    sample_output_adata = sc.read_h5ad(os.path.join(DIR_NAME, "data/annoy_taynaud_15_nn_rs_5.h5ad"))
    sample_cluster_by_obs = sample_output_adata.obs['pheno_louvain'].values.tolist()
    return sample_cluster_by_obs

@pytest.fixture
def r_data():
    with open(os.path.join(DIR_NAME, "data/R_tasic_results.csv")) as csvf:
        myreader = csv.reader(csvf)
        r_list = []
        for row in myreader:
            r_list.append(row[1])
    return r_list[1:]

def test_adata_not_inplace(pca_data, r_data):
    cluster_by_obs, obs_by_cluster, graph, q = cluster_louvain_phenograph(pca_data, 10)
    error_count = 0
    naive_label_map = {}
    for i in range(len(cluster_by_obs)):
        r_val = r_data[i]
        o_val = cluster_by_obs[i]
        if r_val not in naive_label_map:
            naive_label_map[r_val] = o_val
        if naive_label_map[r_val] != o_val:
            error_count += 1
    error = error_count / float(len(cluster_by_obs))
    assert error < .02

def test_adata_inplace(pca_data, r_data):
    cluster_louvain_phenograph(pca_data, 10, annotate=True)
    error_count = 0
    naive_label_map = {}
    for i in range(len(pca_data.obs['pheno_louvain'])):
        r_val = r_data[i]
        o_val = pca_data.obs['pheno_louvain'][i]
        if r_val not in naive_label_map:
            naive_label_map[r_val] = o_val
        if naive_label_map[r_val] != o_val:
            error_count += 1
    error = error_count / float(len(r_data))

    assert error < .02

def test_outlier_label_handling():
    test_adata = MagicMock()
    test_adata.X = None
    with patch.object(tc.clustering, 'phenograph', return_value=([-1, 1, -1, 2, -1, 3], None, None)):
        cluster_by_obs, obs_by_cluster, _, _ = cluster_louvain_phenograph(test_adata, 4, False)
    assert cluster_by_obs == [4, 1, 5, 2, 6, 3]
    assert obs_by_cluster == {1: [1], 4: [0], 5: [2], 2: [3], 6: [4], 3: [5]}

def test_get_annoy_knn_reproducibility(pca_data, sample_graph):
    graph_adata = tc.clustering.get_annoy_knn(pca_data, TEST_K, n_jobs=16, random_seed=TEST_RANDOM_SEED)
    graph_adata_2 = tc.clustering.get_annoy_knn(pca_data, TEST_K, n_jobs=16, random_seed=TEST_RANDOM_SEED)
    print(graph_adata.X.toarray())
    print(graph_adata_2.X.toarray())
    print(sample_graph.X.toarray())

    assert np.array_equal(graph_adata.X.toarray(), graph_adata_2.X.toarray())
    assert False

def test_get_taynaud_louvain_behaviour(sample_graph):
    sample_partition = {i: i % 4 for i in range(284)}
    sample_modularity = 0.6
    expected_cluster_by_obs = [i % 4 for i in range(284)]
    expected_obs_by_cluster = {j: [i * 4 + j for i in range(71)] for j in range(4)}

    with patch.object(tc.clustering.community_louvain, 'best_partition', return_value=sample_partition):
        with patch.object(tc.clustering.community_louvain, 'modularity', return_value=sample_modularity):
            cluster_by_obs, obs_by_cluster, q = tc.clustering.get_taynaud_louvain(sample_graph)

    assert cluster_by_obs == expected_cluster_by_obs
    assert obs_by_cluster == expected_obs_by_cluster
    assert q == sample_modularity

def test_get_taynaud_louvain_reproducibility(sample_graph, annoy_taynaud_sample_partition):
    cluster_by_obs, _, _ = tc.clustering.get_taynaud_louvain(sample_graph, random_seed=TEST_RANDOM_SEED)

    assert cluster_by_obs == annoy_taynaud_sample_partition