import pytest

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from transcriptomic_clustering.utils import get_means_variances_genes
from transcriptomic_clustering.highly_variable_genes import highly_variable_genes, compute_z_scores


@pytest.fixture
def test_matrix():
    """
        test matrix
    """
    mat = np.matrix([[ 9.6650405 ,  0.33695615,  8.68121165,  8.18929134,  8.1016207 ,
         9.15809275,  0.47063873,  9.72234714,  7.16551018,  0.        ],
       [ 6.77953048,  0.        ,  5.9422073 ,  0.        ,  0.        ,
         6.50688935,  6.68938565,  8.35676263,  7.58414682,  0.        ],
       [ 5.60401178,  1.4250022 ,  0.        ,  0.        ,  0.        ,
         4.23403687,  4.85161022, 10.3384316 ,  0.        ,  0.        ],
       [ 5.26653574,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  9.67227696,  8.36744541,  0.        ,  0.        ],
       [ 5.14196888,  1.59518473,  0.        ,  0.        ,  0.        ,
         1.14524834,  0.88740327,  3.23708952,  0.        ,  1.25746609],
       [ 5.09599541,  8.27378078,  0.        ,  0.        ,  0.        ,
         7.25854093,  0.92072549,  9.30755816,  0.        ,  0.        ],
       [ 4.93008834,  0.66350592,  0.        ,  0.        ,  0.        ,
         0.        ,  8.5201187 ,  7.93928909,  0.        ,  0.        ],
       [ 4.69356393,  9.84476317,  0.        ,  0.        ,  0.        ,
         5.03685415,  0.        ,  6.32944125,  0.        ,  0.        ],
       [ 4.46701175,  0.49132133,  0.        ,  0.        ,  0.        ,
         0.35275535,  7.83979371,  9.40005172,  0.        ,  0.19185224],
       [ 4.27977142,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  6.23949258,  7.64510218,  0.        ,  0.21721315]])

    return mat

@pytest.fixture
def test_adata_sparse(test_matrix):
    """
        add test data in AnnData format
    """
    obs_names = ['Gad2_tdTpositive_cell_31', 'Calb2_tdTpositive_cell_62',
                'Htr3a_tdTpositive_cell_27', 'Htr3a_tdTpositive_cell_81',
                'Vip_tdTpositive_cell_61', 'Vip_tdTpositive_cell_5',
                'Htr3a_tdTpositive_cell_57', 'Nos1_tdTpositive_cell_40',
                'Vip_tdTpositive_cell_57', 'Calb2_tdTpositive_cell_25']
    obs = pd.DataFrame(index=obs_names)
    var_names = ['Plp1', 'Npy', 'Cnp', 'Mal', 'Trf', 'Enpp2', 'Penk', 'Cnr1', 'Cd9', 'Rgs5']
    var = pd.DataFrame(index=var_names)

    mat = test_matrix

    adata_sparse = sc.AnnData(X=csr_matrix(mat), obs=obs, var=var)
    adata_sparse.uns['log1p'] = {'base': None}

    return adata_sparse

@pytest.fixture
def test_adata_dense(test_matrix):
    """
        add test data in AnnData format
    """
    obs_names = ['Gad2_tdTpositive_cell_31', 'Calb2_tdTpositive_cell_62',
                'Htr3a_tdTpositive_cell_27', 'Htr3a_tdTpositive_cell_81',
                'Vip_tdTpositive_cell_61', 'Vip_tdTpositive_cell_5',
                'Htr3a_tdTpositive_cell_57', 'Nos1_tdTpositive_cell_40',
                'Vip_tdTpositive_cell_57', 'Calb2_tdTpositive_cell_25']
    obs = pd.DataFrame(index=obs_names)
    var_names = ['Plp1', 'Npy', 'Cnp', 'Mal', 'Trf', 'Enpp2', 'Penk', 'Cnr1', 'Cd9', 'Rgs5']
    var = pd.DataFrame(index=var_names)

    mat = test_matrix

    adata_dense = sc.AnnData(X=mat, obs=obs, var=var)
    adata_dense.uns['log1p'] = {'base': None}

    return adata_dense

@pytest.fixture
def test_dispersion_data():
    """
        test data for compute_z_score
    """
    dispersions = np.array([4.0860877, 4.1462183, 3.6927953, 3.510686, 3.4726, 
                                    3.823604, 3.9517627, 3.9920075, 3.137657, 0.2717548])

    return dispersions


def test_z_score(test_dispersion_data):
    """
        test compute_z_score function
    """
    dispersions = test_dispersion_data

    expected_z_scores = np.array([ 0.95555746,  1.11784422, -0.10590124, -0.59739689, -0.70018737,
                                0.24713897,  0.59302709,  0.70164397, -1.60416661, -9.33896352])

    # test compute_z_scores
    z_scores = compute_z_scores(dispersions)

    np.testing.assert_allclose(
        z_scores,
        expected_z_scores,
        rtol=1e-06,
        atol=1e-06,
    )

def test_get_gene_means_variances_sparse(test_adata_sparse):
    """
        test get_gene_means_variances function with sparse matrix in AnnData
    """

    expected_means = np.array([1783.33005981, 2278.61309844, 626.99537048, 360.11674805,
        329.88117676, 1180.14047895, 2486.76828033, 8472.94624462, 325.87897949, 0.29706002])

    expected_variances = np.array([21743008.95300661, 31907224.71064702,  3090720.77279484, 1167156.65001471,
        979394.31701207,  7862088.7369532 , 22253464.56608047, 83184372.94313633,   447417.83528121, 0.55539122])

    # test get_means_variances_genes
    adata_sparse = test_adata_sparse
    means, variances = get_means_variances_genes(adata = adata_sparse, chunk_size = 5)
    np.testing.assert_allclose(
        means,
        expected_means,
        rtol=1e-06,
        atol=1e-06,
    )

    np.testing.assert_allclose(
        variances,
        expected_variances,
        rtol=1e-06,
        atol=1e-06,
    )

def test_get_gene_means_variances_dense(test_adata_dense):
    """
        test get_gene_means_variances function with dense matrix in AnnData
    """

    expected_means = np.array([1783.33005981, 2278.61309844, 626.99537048, 360.11674805,
        329.88117676, 1180.14047895, 2486.76828033, 8472.94624462, 325.87897949, 0.29706002])

    expected_variances = np.array([21743008.95300661, 31907224.71064702,  3090720.77279484, 1167156.65001471,
        979394.31701207,  7862088.7369532 , 22253464.56608047, 83184372.94313633,   447417.83528121, 0.55539122])

    # test get_means_variances_genes
    adata_dense = test_adata_dense
    means, variances = get_means_variances_genes(adata = adata_dense, chunk_size = 5)
    np.testing.assert_allclose(
        means,
        expected_means,
        rtol=1e-06,
        atol=1e-06,
    )

    np.testing.assert_allclose(
        variances,
        expected_variances,
        rtol=1e-06,
        atol=1e-06,
    )

def test_highly_variable_genes_sparse(test_adata_sparse):
    """
        test highly variable genes of cell expressions with sparse matrix in AnnData
    """
    
    expected_top2_means_log = np.array([7.731761, 7.486798])
    expected_top2_dispersions_log = np.array([9.547092, 9.408647])

    expected_hvg = ['Npy','Plp1']

    adata = test_adata_sparse

    # test highly_variable_genes
    means, variances = get_means_variances_genes(adata = adata, chunk_size = 5)
    highly_variable_genes(adata = adata, means = means, variances = variances, max_genes=2)

    np.testing.assert_array_equal(
        np.sort(adata.var_names[adata.var['highly_variable']]),
        np.sort(expected_hvg),
    )

    np.testing.assert_allclose(
        np.sort(adata.var['means_log'][adata.var['highly_variable']]),
        np.sort(expected_top2_means_log),
        rtol=1e-06,
        atol=1e-06,
    )

    np.testing.assert_allclose(
        np.sort(adata.var['dispersions_log'][adata.var['highly_variable']]),
        np.sort(expected_top2_dispersions_log),
        rtol=1e-06,
        atol=1e-06,
    )

def test_highly_variable_genes_dense(test_adata_dense):
    """
        test highly variable genes of cell expressions with dense matrix in AnnData
    """
    
    expected_top2_means_log = np.array([7.731761, 7.486798])
    expected_top2_dispersions_log = np.array([9.547092, 9.408647])

    expected_hvg = ['Npy','Plp1']

    adata = test_adata_dense

    # test highly_variable_genes
    means, variances = get_means_variances_genes(adata = adata, chunk_size = 5)
    highly_variable_genes(adata = adata, means = means, variances = variances, max_genes=2)

    np.testing.assert_array_equal(
        np.sort(adata.var_names[adata.var['highly_variable']]),
        np.sort(expected_hvg),
    )

    np.testing.assert_allclose(
        np.sort(adata.var['means_log'][adata.var['highly_variable']]),
        np.sort(expected_top2_means_log),
        rtol=1e-06,
        atol=1e-06,
    )

    np.testing.assert_allclose(
        np.sort(adata.var['dispersions_log'][adata.var['highly_variable']]),
        np.sort(expected_top2_dispersions_log),
        rtol=1e-06,
        atol=1e-06,
    )

