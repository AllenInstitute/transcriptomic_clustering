import pytest
import os
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix
import scanpy as sc

from transcriptomic_clustering.normalize import (
    normalize, log1p_of_cpm)


def test_normalization():
    """
        test normalization of cell expressions with a dense & sparse matrices
    """
    arr = np.array([[ 6,  6,  4,  2, 18],
                    [ 1,  0,  4,  8, 11],
                    [ 6, 11,  3, 26, 26],
                    [ 0,  0, 19,  9,  0]])

    expected_norm = np.array([[12.023757, 12.023757, 11.618295, 10.925157, 13.122365],
                            [10.637481,  0.      , 12.023757, 12.716901, 13.035354],
                            [11.330616, 11.936747, 10.637481, 12.796944, 12.796944],
                            [ 0.      ,  0.      , 13.427747, 12.680533,  0.      ]])

    # dense matrix
    anndata_dense = sc.AnnData(arr)
    result_dense = normalize(anndata_dense)

    np.testing.assert_almost_equal(result_dense.X, expected_norm, decimal = 6)

    # sparse matrix
    anndata_sparse = sc.AnnData(arr)
    anndata_sparse.X = csr_matrix(anndata_sparse.X)
    result_sparse = normalize(anndata_sparse)
    result_sparse.X = csr_matrix.toarray(result_sparse.X)
    
    np.testing.assert_almost_equal(result_sparse.X, expected_norm, decimal = 6)


def test_normalize():
    """
        test normalization of cell expressions with a dense & sparse matrices
    """
    arr = np.array([[6, 6, 4, 2, 18],
                    [1, 0, 4, 8, 11],
                    [6, 11, 3, 26, 26],
                    [0, 0, 19, 9, 0]])

    expected_norm = np.array([[17.3466147, 17.3466147, 16.7616566, 15.7616695, 18.9315715],
                              [15.3466407, 0.0000000, 17.3466147, 18.3466104, 18.8060408],
                              [16.3466234, 17.2210846, 15.3466407, 18.4620873, 18.4620873],
                              [0.0000000, 0.0000000, 19.3721433, 18.2941431, 0.0000000]])

    # dense matrix
    anndata_dense = ad.AnnData(arr)
    result_dense = log1p_of_cpm(anndata_dense, inplace=False)
    print(result_dense.X)
    np.testing.assert_almost_equal(result_dense.X, expected_norm, decimal=6)

    # sparse matrix
    anndata_sparse = ad.AnnData(arr)
    anndata_sparse.X = csr_matrix(anndata_sparse.X)
    result_sparse = log1p_of_cpm(anndata_sparse, inplace=False)
    result_sparse.X = csr_matrix.toarray(result_sparse.X)

#    assert False
    np.testing.assert_almost_equal(result_sparse.X, expected_norm, decimal=6)


def test_backed_normalize():
    test_dir ="/home/sergeyg/repos/transcriptomic_clustering/notebooks/write/test/"
    input_file_name = os.path.join(test_dir, "input10x4_csr.h5ad")
    expected_output_file_name = os.path.join(test_dir, "log1p_of_cpm_10x4_csr.h5ad")
    test_output_file_name = os.path.join(test_dir, "test_log1p_of_cpm_10x4_csr.h5ad")

    expected = sc.read_h5ad(expected_output_file_name)

    adata = sc.read_h5ad(input_file_name, backed='r')
    print(f'loaded {input_file_name}')
    obtained = log1p_of_cpm(adata, inplace=False, filename=test_output_file_name, chunk_size=5)

    obtained = sc.read_h5ad(test_output_file_name)
    print(obtained.X)
    print(expected.X)

    np.testing.assert_almost_equal(csr_matrix.toarray(obtained.X), csr_matrix.toarray(expected.X), decimal=5)
