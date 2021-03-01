import pytest

import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix

from transcriptomic_clustering.normalize import normalize_cell_expresions

arr = np.array([[ 6,  6,  4,  2, 18],
                [ 1,  0,  4,  8, 11],
                [ 6, 11,  3, 26, 26],
                [ 0,  0, 19,  9,  0]])

expected_norm = np.array([[17.3466147, 17.3466147, 16.7616566, 15.7616695, 18.9315715],
                    [15.3466407,  0.0000000, 17.3466147, 18.3466104, 18.8060408],
                    [16.3466234, 17.2210846, 15.3466407, 18.4620873, 18.4620873],
                    [ 0.0000000,  0.0000000, 19.3721433, 18.2941431,  0.0000000]])

def test_normalization_dense_matrix():
    """
        test normalization of cell expressions with a dense matrix
    """

    a_data = ad.AnnData(arr)
    result = normalize_cell_expresions(a_data)

    np.testing.assert_almost_equal(result.X, expected_norm, decimal = 6)

def test_normalization_csr_matrix():
    """
        test normalization of cell expressions with a sparse csr_matrix
    """

    a_data = ad.AnnData(arr)
    a_data.X = csr_matrix(a_data.X)
    result = normalize_cell_expresions(a_data)
    result.X = csr_matrix.toarray(result.X)
    
    np.testing.assert_almost_equal(result.X, expected_norm, decimal = 6)