import pytest

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

import transcriptomic_clustering as tc
from transcriptomic_clustering.highly_variable_genes import compute_z_scores


@pytest.fixture
def test_matrix():
    """
        test matrix
    """
    mat = np.array([[ 9.66504   ,  0.33695614,  8.681211  ,  8.189291  ,  8.101621  ,
         9.1580925 ,  0.47063872,  9.722347  ,  7.1655097 ,  0.        ,
         0.        ,  5.66686   ,  5.2936516 ,  3.6814382 ,  3.6931765 ,
         7.20932   ,  5.184904  ,  0.        ,  0.12771612,  7.304134  ],
       [ 6.7795305 ,  0.        ,  5.9422073 ,  0.        ,  0.        ,
         6.5068893 ,  6.6893854 ,  8.356763  ,  7.584147  ,  0.        ,
         0.        ,  6.891058  ,  0.32603756,  3.7383661 ,  1.6596369 ,
         6.0409465 ,  5.59895   ,  0.        ,  0.        ,  6.711624  ],
       [ 5.6040115 ,  1.4250022 ,  0.        ,  0.        ,  0.        ,
         4.234037  ,  4.85161   , 10.338431  ,  0.        ,  0.        ,
         0.        ,  8.530924  ,  0.        ,  7.2313104 ,  5.873241  ,
         5.432499  ,  5.6369367 ,  1.4265201 ,  2.8672924 ,  9.387415  ],
       [ 5.2665358 ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  9.672277  ,  8.367446  ,  0.        ,  0.        ,
         6.0757575 ,  6.9293156 ,  2.0590003 ,  0.        ,  2.2087634 ,
         5.3162007 ,  7.3012366 ,  0.67047584,  0.03554144,  7.964448  ],
       [ 5.1419687 ,  1.5951848 ,  0.        ,  0.        ,  0.        ,
         1.1452483 ,  0.8874033 ,  3.2370896 ,  0.        ,  1.2574661 ,
         0.        ,  6.762831  ,  4.5374804 ,  5.635769  ,  0.        ,
         6.4214377 ,  6.559067  ,  0.        ,  0.        ,  8.028691  ],
       [ 5.0959954 ,  8.273781  ,  0.        ,  0.        ,  0.        ,
         7.258541  ,  0.92072546,  9.307558  ,  0.        ,  0.        ,
         0.        ,  6.7757244 ,  5.08293   ,  5.439114  ,  1.8826606 ,
         5.852033  ,  5.794861  ,  0.        ,  0.        ,  7.664306  ],
       [ 4.9300885 ,  0.6635059 ,  0.        ,  0.        ,  0.        ,
         0.        ,  8.520119  ,  7.939289  ,  0.        ,  0.        ,
         0.        ,  6.8099427 ,  6.245945  ,  4.177312  ,  1.6467462 ,
         4.5364413 ,  0.        ,  0.        ,  0.16657794,  7.713487  ],
       [ 4.693564  ,  9.844763  ,  0.        ,  0.        ,  0.        ,
         5.0368543 ,  0.        ,  6.329441  ,  0.        ,  0.        ,
         0.        ,  7.3114924 ,  0.        ,  5.2191944 ,  5.429897  ,
         5.8731956 ,  6.385457  ,  0.        ,  0.15010178,  7.895425  ],
       [ 4.467012  ,  0.49132133,  0.        ,  0.        ,  0.        ,
         0.35275534,  7.8397937 ,  9.400052  ,  0.        ,  0.19185223,
         0.        ,  6.892822  ,  0.4202775 ,  6.145297  ,  0.3718924 ,
         6.398558  ,  5.6159244 ,  0.        ,  0.        ,  7.8201156 ],
       [ 4.2797713 ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  6.239493  ,  7.645102  ,  0.        ,  0.21721315,
         0.        ,  7.660991  ,  5.539333  ,  0.        ,  1.317859  ,
         6.317706  ,  5.6816654 ,  0.        ,  0.        ,  7.363291  ]])

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
    var_names = ['Plp1', 'Npy', 'Cnp', 'Mal', 'Trf', 'Enpp2', 'Penk', 'Cnr1', 'Cd9', 'Rgs5',
            'mt_AK131586', 'mt_AK138996', 'mt_AK139026', 'mt_AK140457', 'mt_AK157367',
            'mt_AK159262', 'mt_AK162543', 'mt_AK201028', 'mt_BC006023', 'mt_GU332589']
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
    var_names = ['Plp1', 'Npy', 'Cnp', 'Mal', 'Trf', 'Enpp2', 'Penk', 'Cnr1', 'Cd9', 'Rgs5',
            'mt_AK131586', 'mt_AK138996', 'mt_AK139026', 'mt_AK140457', 'mt_AK157367',
            'mt_AK159262', 'mt_AK162543', 'mt_AK201028', 'mt_BC006023', 'mt_GU332589']
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
        test get_means_vars_genes function with sparse matrix in AnnData
    """

    expected_means = np.array([1783.32848206, 2278.61309851, 1180.14047895, 2486.7697696,
                    8472.94668407, 1460.81580505,  122.69405152,  268.27713737,
                    64.60056841,  476.14386215,  439.80124054, 3130.15311279])

    expected_variances = np.array([24158850.37172627, 35452471.90035736,  8735654.15217023,
                    24726115.19952307, 92427076.976327, 1831559.06557216,
                    28041.36235863, 175094.93196007, 15236.32455233,
                    124907.24296245, 172951.41672484, 10052863.13058035])


    # test get_means_vars_genes
    adata_sparse = test_adata_sparse
    means, variances, gene_mask = tc.get_means_vars_genes(adata = adata_sparse)
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
        test get_means_vars_genes function with dense matrix in AnnData
    """

    expected_means = np.array([1783.32848206, 2278.61309851, 1180.14047895, 2486.7697696,
                    8472.94668407, 1460.81580505,  122.69405152,  268.27713737,
                    64.60056841,  476.14386215,  439.80124054, 3130.15311279])

    expected_variances = np.array([24158850.37172627, 35452471.90035736,  8735654.15217023,
                    24726115.19952307, 92427076.976327, 1831559.06557216,
                    28041.36235863, 175094.93196007, 15236.32455233,
                    124907.24296245, 172951.41672484, 10052863.13058035])

    # test get_means_vars_genes
    adata_dense = test_adata_dense
    means, variances, gene_mask = tc.get_means_vars_genes(adata = adata_dense)
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

def test_gene_mask(test_adata_sparse):
    """
        test gene mask with sparse matrix in AnnData
    """

    expected_gene_mask = [True, True, False, False, False, True, True, True, False, False, 
                    False, True, True, True, True, True, True, False, False, True]
    
    adata = test_adata_sparse
    means, variances, gene_mask = tc.get_means_vars_genes(adata = adata, chunk_size = 5)

    np.testing.assert_array_equal(
        gene_mask,
        expected_gene_mask,
    )

def test_gene_mask(test_adata_dense):
    """
        test gene mask with dense matrix in AnnData
    """

    expected_gene_mask = [True, True, False, False, False, True, True, True, False, False, 
                    False, True, True, True, True, True, True, False, False, True]
    
    adata = test_adata_dense
    means, variances, gene_mask = tc.get_means_vars_genes(adata = adata, chunk_size = 5)

    np.testing.assert_array_equal(
        gene_mask,
        expected_gene_mask,
    )

def test_highly_variable_genes_sparse(test_adata_sparse):
    """
        test highly variable genes of cell expressions with sparse matrix in AnnData
    """

    expected_hvg = ['Npy','Plp1']

    adata = test_adata_sparse

    # test highly_variable_genes
    means, variances, gene_mask = tc.get_means_vars_genes(adata = adata, chunk_size = 5)
    tc.highly_variable_genes(adata = adata, means = means, variances = variances, gene_mask=gene_mask, max_genes=2)

    np.testing.assert_array_equal(
        np.sort(adata.var_names[adata.var['highly_variable']]),
        np.sort(expected_hvg),
    )

def test_highly_variable_genes_dense(test_adata_dense):
    """
        test highly variable genes of cell expressions with dense matrix in AnnData
    """

    expected_hvg = ['Npy','Plp1']

    adata = test_adata_dense

    # test highly_variable_genes
    means, variances, gene_mask = tc.get_means_vars_genes(adata = adata, chunk_size = 5)
    tc.highly_variable_genes(adata = adata, means = means, variances = variances, gene_mask=gene_mask, max_genes=2)

    np.testing.assert_array_equal(
        np.sort(adata.var_names[adata.var['highly_variable']]),
        np.sort(expected_hvg),
    )

