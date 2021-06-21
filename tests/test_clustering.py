from annoy import AnnoyIndex
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
def sample_index_file():
    return os.path.join(DIR_NAME, "data/sample_annoy_index.ann")

@pytest.fixture
def annoy_taynaud_sample_partition():
    sample_output_adata = sc.read_h5ad(os.path.join(DIR_NAME, "data/annoy_taynaud_15_nn_rs_5.h5ad"))
    sample_cluster_by_obs = sample_output_adata.obs['pheno_louvain'].values.tolist()
    return sample_cluster_by_obs

@pytest.fixture
def annoy_vtraag_sample_partition():
    sample_output_adata = sc.read_h5ad(os.path.join(DIR_NAME, "data/annoy_vtraag_15_nn_rs_5.h5ad"))
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

def test_phenograph_adata_not_inplace(pca_data, r_data):
    _, _, _, q = cluster_louvain_phenograph(pca_data, 10)
    assert q > .8

def test_phenograph_adata_inplace(pca_data, r_data):
    cluster_by_obs, _, _, q = cluster_louvain_phenograph(pca_data, 10, annotate=True)

    print(f"Cluster by obs has type {type(cluster_by_obs)}")
    print(f"pca obs has type {type(pca_data.obs['pheno_louvain'])}")
    assert cluster_by_obs.tolist() == pca_data.obs['pheno_louvain'].to_list()
    assert q > .8

def test_phenograph_outlier_label_handling():
    test_adata = MagicMock()
    test_adata.X = None
    with patch.object(tc.clustering, 'phenograph', return_value=([-1, 1, -1, 2, -1, 3], None, None)):
        cluster_by_obs, obs_by_cluster, _, _ = cluster_louvain_phenograph(test_adata, 4, False)
    assert cluster_by_obs == [4, 1, 5, 2, 6, 3]
    assert obs_by_cluster == {1: [1], 4: [0], 5: [2], 2: [3], 6: [4], 3: [5]}

def test_annoy_build_csr_nn_graph_reproducibility(pca_data, sample_graph, sample_index_file):
    graph_csr = tc.clustering._annoy_build_csr_nn_graph(pca_data, sample_index_file, TEST_K)
    assert np.array_equal(sample_graph.X.toarray(), graph_csr.toarray().astype('float32'))

def test_annoy_build_csr_nn_graph_reproducibility_uniform(pca_data, sample_graph, sample_index_file):
    uniform_sample_graph_arr = (sample_graph.X.toarray() > 0).astype('float32')
    graph_csr = tc.clustering._annoy_build_csr_nn_graph(pca_data, sample_index_file, TEST_K, weighting_method='uniform')
    assert np.array_equal(uniform_sample_graph_arr, graph_csr.toarray())

def test_annoy_build_csr_nn_graph_reproducibility_multithread(pca_data, sample_graph, sample_index_file):
    graph_csr = tc.clustering._annoy_build_csr_nn_graph(pca_data, sample_index_file, TEST_K, n_jobs=8)
    assert np.array_equal(sample_graph.X.toarray(), graph_csr.toarray().astype('float32'))

def test_annoy_build_csr_nn_graph_bad_weighting(pca_data, sample_index_file):
    with pytest.raises(ValueError) as err:
        graph_csr = tc.clustering._annoy_build_csr_nn_graph(pca_data, sample_index_file, TEST_K, weighting_method='fake method')
    assert str(err.value) == 'fake method is not a valid weighting option! Must use jaccard or uniform'

def test_get_annoy_knn_new_index(pca_data, sample_graph, sample_index_file):
    graph_adata = tc.clustering.get_annoy_knn(pca_data, TEST_K, random_seed=TEST_RANDOM_SEED)
    assert np.array_equal(sample_graph.X.toarray(), graph_adata.X.toarray())

def test_get_annoy_knn_reproducibility(pca_data, sample_graph, sample_index_file):
    graph_adata = tc.clustering.get_annoy_knn(pca_data, TEST_K, random_seed=TEST_RANDOM_SEED, annoy_index_filename=sample_index_file)
    assert np.array_equal(sample_graph.X.toarray(), graph_adata.X.toarray())

def test_get_annoy_knn_reproducibility_multi_thread(pca_data, sample_graph, sample_index_file):
    graph_adata = tc.clustering.get_annoy_knn(pca_data, TEST_K, n_jobs=8, random_seed=TEST_RANDOM_SEED, annoy_index_filename=sample_index_file)
    assert np.array_equal(sample_graph.X.toarray(), graph_adata.X.toarray())

def test_get_taynaud_louvain_behaviour(sample_graph):
    sample_partition = {i: i % 4 for i in range(284)}
    sample_modularity = 0.6
    expected_cluster_by_obs = [i % 4 for i in range(284)]

    with patch.object(tc.clustering.community_louvain, 'best_partition', return_value=sample_partition):
        with patch.object(tc.clustering.community_louvain, 'modularity', return_value=sample_modularity):
            cluster_by_obs, q = tc.clustering.get_taynaud_louvain(sample_graph)

    assert cluster_by_obs == expected_cluster_by_obs
    assert q == sample_modularity

def test_get_taynaud_louvain_reproducibility(sample_graph, annoy_taynaud_sample_partition):
    cluster_by_obs, _ = tc.clustering.get_taynaud_louvain(sample_graph, random_seed=TEST_RANDOM_SEED)
    assert cluster_by_obs == annoy_taynaud_sample_partition

def test_get_vtraag_leiden_reproducibility(sample_graph, annoy_vtraag_sample_partition):
    cluster_by_obs, _ = tc.clustering.get_vtraag_leiden(sample_graph, random_seed=TEST_RANDOM_SEED)
    assert cluster_by_obs == annoy_vtraag_sample_partition