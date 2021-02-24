import unittest

import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix

from transcriptomic_clustering.normalization.normalize import normalize_cell_expresions

class TestNormalization(unittest.TestCase):
    
    def test_normalization(self):
        """
            test normalization of cell expressions with a small example
        """
        arr = np.array([[ 6,  6,  4,  2, 18],
                        [ 1,  0,  4,  8, 11],
                        [ 6, 11,  3, 26, 26],
                        [ 0,  0, 19,  9,  0]])

        arr_norm = np.array([[17.35, 17.35, 16.76, 15.76, 18.93],
                            [15.35,  0.  , 17.35, 18.35, 18.81],
                            [16.35, 17.22, 15.35, 18.46, 18.46],
                            [ 0. ,  0. , 19.37, 18.29,  0. ]])

        a_data = ad.AnnData(arr)
        a_data.X = csr_matrix(a_data.X)
        result = normalize_cell_expresions(a_data)
        result.X = csr_matrix.toarray(result.X)
        self.assertAlmostEqual(result.X.all(), arr_norm.all())