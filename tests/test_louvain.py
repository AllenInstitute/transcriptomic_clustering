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
    return r_list

def test_array_not_inplace(pca_data, r_data):
    communities, graph, q = cluster_louvain(pca_data.X, 10)
    assert r_data == communities