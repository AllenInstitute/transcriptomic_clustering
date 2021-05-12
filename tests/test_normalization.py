import os
import pytest
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
import anndata as ad

from transcriptomic_clustering.normalization import (
    normalize, normalize_backed, normalize_inmemory)


@pytest.fixture
def X():
    return np.array([[ 6,  6,  4,  2, 18],
                    [ 1,  0,  4,  8, 11],
                    [ 6, 11,  3, 26, 26],
                    [ 0,  0, 19,  9,  0]])


@pytest.fixture
def normalized_X():

    normalized = np.array([[12.023757, 12.023757, 11.618295, 10.925157, 13.122365],
                            [10.637481,  0.      , 12.023757, 12.716901, 13.035354],
                            [11.330616, 11.936747, 10.637481, 12.796944, 12.796944],
                            [ 0.      ,  0.      , 13.427747, 12.680533,  0.      ]])

    return normalized


def test_normalize_in_memory(X, normalized_X):

    expected = normalized_X

    # dense matrix:
    adata = ad.AnnData(X)
    obtained = normalize_inmemory(adata, inplace=False).X

    np.testing.assert_almost_equal(obtained, expected, decimal=6)


    # sparse scr matrix:
    adata = ad.AnnData(csr_matrix(X))
    obtained = normalize_inmemory(adata, inplace=False).X

    np.testing.assert_almost_equal(csr_matrix.toarray(obtained), expected, decimal=6)


def test_normalize_backed(X, normalized_X, tmpdir_factory):

    expected_X = normalized_X

    tmpdir = str(tmpdir_factory.mktemp("test_normalize"))
    input_file_name = os.path.join(tmpdir, "input.h5ad")
    output_file_name = os.path.join(tmpdir, "output.h5ad")

    ad.AnnData(csr_matrix(X)).write(input_file_name) # make tmp input file

    adata = sc.read_h5ad(input_file_name, backed='r')
    adata_normalized = normalize_backed(adata, copy_to=output_file_name, chunk_size=2)
    adata_normalized.file.close()

    obtained_adata = sc.read_h5ad(output_file_name) # read normalized data
    obtained_X = obtained_adata.X

    np.testing.assert_almost_equal(csr_matrix.toarray(obtained_X), expected_X, decimal=6)
    assert obtained_adata.n_obs == 4
    assert obtained_adata.n_vars == 5


def test_normalize_inplace(X):

    adata = ad.AnnData(X)

    # returning new object
    adata_normalized = normalize(adata, inplace=False)
    assert adata_normalized is not adata

    # return same object
    adata_normalized = normalize(adata, inplace=True)
    assert adata_normalized is adata

