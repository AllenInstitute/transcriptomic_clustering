import pytest

import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix

from transcriptomic_clustering.normalize import normalize_cell_expresions


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
    result_dense = normalize_cell_expresions(anndata_dense)

    np.testing.assert_almost_equal(result_dense.X, expected_norm, decimal = 6)

    # sparse matrix
    anndata_sparse = sc.AnnData(arr)
    anndata_sparse.X = csr_matrix(anndata_sparse.X)
    result_sparse = normalize_cell_expresions(anndata_sparse)
    result_sparse.X = csr_matrix.toarray(result_sparse.X)
    
    np.testing.assert_almost_equal(result_sparse.X, expected_norm, decimal = 6)