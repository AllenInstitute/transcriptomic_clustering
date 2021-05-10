import scanpy as sc
import pytest
import csv

from transcriptomic_clustering.louvain import cluster_louvain

@pytest.fixture
def pca_data():
    return sc.read_csv("data/output_dimreduction_pca.csv", first_column_names=True)

@pytest.fixture
def r_data():
    with open("data/R_tasic_results.csv") as csvf:
        myreader = csv.reader(csvf)
        r_list = []
        for row in myreader:
            r_list.append(row[1])
    return r_list[1:]

def test_array_not_inplace(pca_data, r_data):
    communities, graph, q = cluster_louvain(pca_data.X, 10)
    error_count = 0
    naive_label_map = {}
    for i in range(len(communities)):
        r_val = r_data[i]
        o_val = communities[i]
        if r_val not in naive_label_map:
            naive_label_map[r_val] = o_val
        if naive_label_map[r_val] != o_val:
            error_count += 1
    error = error_count / float(len(communities))
    assert error < .05

def test_adata_not_inplace(pca_data, r_data):
    communities, graph, q = cluster_louvain(pca_data, 10)
    error_count = 0
    naive_label_map = {}
    for i in range(len(communities)):
        r_val = r_data[i]
        o_val = communities[i]
        if r_val not in naive_label_map:
            naive_label_map[r_val] = o_val
        if naive_label_map[r_val] != o_val:
            error_count += 1
    error = error_count / float(len(communities))
    assert error < .05

def test_adata_inplace(pca_data, r_data):
    cluster_louvain(pca_data, 10, inplace=True)
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
    assert error < .05