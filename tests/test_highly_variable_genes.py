import pytest

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

from transcriptomic_clustering.select_highly_variable_genes import select_highly_variable_genes, compute_z_scores


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

    mat = np.array([[15756.00804387,     0.40067764,  5890.18033216,  3601.16861741,
         3298.81173264,  9489.93824922,     0.60101648, 16685.36398061,
         1293.02161447,     0.        ],
       [  878.65560937,     0.        ,   379.77448518,     0.        ,
            0.        ,   668.73984918,   802.82826638,  4257.88482616,
         1965.76790188,     0.        ],
       [  270.51347699,     3.15786697,     0.        ,     0.        ,
            0.        ,    67.99519561,   126.94624525, 30896.53233773,
            0.        ,     0.        ],
       [  192.74362096,     0.        ,     0.        ,     0.        ,
            0.        ,     0.        , 15870.44651329,  4303.62545797,
            0.        ,     0.        ],
       [  170.05221786,     3.92923957,     0.        ,     0.        ,
            0.        ,     2.14322185,     1.42881449,    24.45951445,
            0.        ,     2.51649971],
       [  162.36637952,  3918.74068774,     0.        ,     0.        ,
            0.        ,  1419.18287479,     1.51111152, 11020.00371581,
            0.        ,     0.        ],
       [  137.3917374 ,     0.94158747,     0.        ,     0.        ,
            0.        ,     0.        ,  5013.64897325,  2804.36544014,
            0.        ,     0.        ],
       [  108.2418166 , 18858.33257491,     0.        ,     0.        ,
            0.        ,   152.98483866,     0.        ,   559.84313403,
            0.        ,     0.        ],
       [   86.09606897,     0.63447447,     0.        ,     0.        ,
            0.        ,     0.42298297,  2538.68085668, 12088.00595288,
            0.        ,     0.21149149],
       [   71.22392899,     0.        ,     0.        ,     0.        ,
            0.        ,     0.        ,   511.59834128,  2089.38215702,
            0.        ,     0.24260894]])

    adata = sc.AnnData(X=csr_matrix(mat), obs=obs, var=var)
    ad_dense = sc.AnnData(X=mat, obs=obs, var=var)
    
    # expected results to be compared
    dispersions = np.array([4.0860877, 4.1462183, 3.6927953, 3.510686, 3.4726, 
                                    3.823604, 3.9517627, 3.9920075, 3.137657, 0.2717548])

    expected_z_scores = np.array([ 0.95555746,  1.11784422, -0.10590124, -0.59739689, -0.70018737,
                                0.24713897,  0.59302709,  0.70164397, -1.60416661, -9.33896352])

    expected_top2_means_log = np.array([7.731761, 7.486798])
    expected_top2_dispersions_log = np.array([9.652446, 9.513999])

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
    select_highly_variable_genes(adata = adata, max_genes=2)

    np.testing.assert_array_equal(
        np.sort(adata.var_names),
        np.sort(expected_hvg),
    )

    np.testing.assert_allclose(
        np.sort(adata.var['means_log']),
        np.sort(expected_top2_means_log),
        rtol=1e-06,
        atol=1e-06,
    )

    np.testing.assert_allclose(
        np.sort(adata.var['dispersions_log']),
        np.sort(expected_top2_dispersions_log),
        rtol=1e-06,
        atol=1e-06,
    )

    # test dense matrix case
    select_highly_variable_genes(adata = ad_dense, max_genes=2)

    np.testing.assert_array_equal(
        np.sort(ad_dense.var_names),
        np.sort(expected_hvg),
    )

    np.testing.assert_allclose(
        np.sort(ad_dense.var['means_log']),
        np.sort(expected_top2_means_log),
        rtol=1e-06,
        atol=1e-06,
    )

    np.testing.assert_allclose(
        np.sort(ad_dense.var['dispersions_log']),
        np.sort(expected_top2_dispersions_log),
        rtol=1e-06,
        atol=1e-06,
    )

