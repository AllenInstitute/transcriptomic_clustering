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

    mat = np.array([[13.943706  ,  0.48612496, 12.524341  , 11.81465   , 11.688168  ,
         13.212335  ,  0.67898816, 14.026382  , 10.337646  ,  0.        ],
        [ 9.780795  ,  0.        ,  8.572793  ,  0.        ,  0.        ,
          9.387457  ,  9.6507435 , 12.05626   , 10.941611  ,  0.        ],
        [ 8.08488   ,  2.0558436 ,  0.        ,  0.        ,  0.        ,
          6.108424  ,  6.999394  , 14.915204  ,  0.        ,  0.        ],
        [ 7.598005  ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        , 13.954146  , 12.071672  ,  0.        ,  0.        ],
        [ 7.418293  ,  2.3013651 ,  0.        ,  0.        ,  0.        ,
          1.6522441 ,  1.2802523 ,  4.670133  ,  0.        ,  1.8141401 ],
        [ 7.3519673 , 11.9365425 ,  0.        ,  0.        ,  0.        ,
         10.471861  ,  1.3283261 , 13.427968  ,  0.        ,  0.        ],
        [ 7.112614  ,  0.9572367 ,  0.        ,  0.        ,  0.        ,
          0.        , 12.291933  , 11.453973  ,  0.        ,  0.        ],
        [ 6.7713814 , 14.202991  ,  0.        ,  0.        ,  0.        ,
          7.2666445 ,  0.        ,  9.1314535 ,  0.        ,  0.        ],
        [ 6.4445357 ,  0.70882684,  0.        ,  0.        ,  0.        ,
          0.5089184 , 11.3104315 , 13.561408  ,  0.        ,  0.27678427],
        [ 6.174405  ,  0.        ,  0.        ,  0.        ,  0.        ,
          0.        ,  9.001685  , 11.029551  ,  0.        ,  0.31337234]])

    adata = sc.AnnData(X=csr_matrix(mat), obs=obs, var=var)
    
    # expected results to be compared
    expected_means = np.array([1783.3291, 2278.613,  626.99536,  360.11676,  329.88107,
                            1180.1404, 2486.7698 , 8472.947,  325.87897, 0.29706])
    
    expected_dispersions = np.array([4.0860877, 4.1462183, 3.6927953, 3.510686, 3.4726, 
                                    3.823604, 3.9517627, 3.9920075, 3.137657, 0.2717548])

    expected_z_scores = np.array([ 0.95555746,  1.11784422, -0.10590124, -0.59739689, -0.70018737,
                                0.24713897,  0.59302709,  0.70164397, -1.60416661, -9.33896352])

    expected_top2_means = np.array([1783.329102, 2278.613037])
    expected_top2_dispersions = np.array([4.086088, 4.146218])

    expected_hvg = ['Plp1', 'Npy']

    # test compute_z_scores
    z_scores = compute_z_scores(expected_dispersions)

    np.testing.assert_allclose(
        z_scores,
        expected_z_scores,
        rtol=1e-06,
        atol=1e-06,
    )

    # test select_highly_variable_genes
    select_highly_variable_genes(ad_norm = adata, low_thresh = 0, min_cells = 1, max_genes=2)

    np.testing.assert_array_equal(
        np.sort(adata.var_names),
        np.sort(expected_hvg),
    )

    np.testing.assert_allclose(
        np.sort(adata.var['means']),
        np.sort(expected_top2_means),
        rtol=1e-06,
        atol=1e-06,
    )

    np.testing.assert_allclose(
        np.sort(adata.var['dispersions']),
        np.sort(expected_top2_dispersions),
        rtol=1e-06,
        atol=1e-06,
    )


