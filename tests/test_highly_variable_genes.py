import pytest

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from transcriptomic_clustering.utils import get_gene_means_variances
from transcriptomic_clustering.highly_variable_genes import highly_variable_genes, compute_z_scores


def test_highly_variable_genes():
    """
        test highly variable genes of cell expressions
    """
    
    # prepare dataset
    obs_names = ['Gad2_tdTpositive_cell_31', 'Calb2_tdTpositive_cell_62',
                'Htr3a_tdTpositive_cell_27', 'Htr3a_tdTpositive_cell_81',
                'Vip_tdTpositive_cell_61', 'Vip_tdTpositive_cell_5',
                'Htr3a_tdTpositive_cell_57', 'Nos1_tdTpositive_cell_40',
                'Vip_tdTpositive_cell_57', 'Calb2_tdTpositive_cell_25']
    obs = pd.DataFrame(index=obs_names)
    var_names = ['Plp1', 'Npy', 'Cnp', 'Mal', 'Trf', 'Enpp2', 'Penk', 'Cnr1', 'Cd9', 'Rgs5']
    var = pd.DataFrame(index=var_names)

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

    adata = sc.AnnData(X=csr_matrix(mat), obs=obs, var=var)
    adata.uns['log1p'] = {'base': None}
    ad_dense = sc.AnnData(X=mat, obs=obs, var=var)
    ad_dense.uns['log1p'] = {'base': None}

    # expected results to be compared
    dispersions = np.array([4.0860877, 4.1462183, 3.6927953, 3.510686, 3.4726, 
                                    3.823604, 3.9517627, 3.9920075, 3.137657, 0.2717548])

    expected_z_scores = np.array([ 0.95555746,  1.11784422, -0.10590124, -0.59739689, -0.70018737,
                                0.24713897,  0.59302709,  0.70164397, -1.60416661, -9.33896352])

    expected_top2_means_log = np.array([7.731761, 7.486798])
    expected_top2_dispersions_log = np.array([9.547092, 9.408647])

    expected_hvg = ['Npy','Plp1']

    # test compute_z_scores
    z_scores = compute_z_scores(dispersions)

    np.testing.assert_allclose(
        z_scores,
        expected_z_scores,
        rtol=1e-06,
        atol=1e-06,
    )

    # test select_highly_variable_genes
    means, variances = get_gene_means_variances(adata = adata, chunk_size = 5)
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

    # test dense matrix case
    means, variances = get_gene_means_variances(adata = ad_dense, chunk_size = 5)
    highly_variable_genes(adata = ad_dense, means = means, variances = variances, max_genes=2)

    np.testing.assert_array_equal(
        np.sort(ad_dense.var_names[adata.var['highly_variable']]),
        np.sort(expected_hvg),
    )

    np.testing.assert_allclose(
        np.sort(ad_dense.var['means_log'][adata.var['highly_variable']]),
        np.sort(expected_top2_means_log),
        rtol=1e-06,
        atol=1e-06,
    )

    np.testing.assert_allclose(
        np.sort(ad_dense.var['dispersions_log'][adata.var['highly_variable']]),
        np.sort(expected_top2_dispersions_log),
        rtol=1e-06,
        atol=1e-06,
    )

