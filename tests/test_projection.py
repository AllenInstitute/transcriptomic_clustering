import os
import pytest

import numpy as np
import scipy as scp
import pandas as pd
import scanpy as sc
import anndata as ad
import transcriptomic_clustering as tc


@pytest.fixture
def test_adata():
    rng = np.random.default_rng(1)
    adata = ad.AnnData(rng.random((20,10)))
    return adata


def test_simple_proj(test_adata):
    X_expected = -test_adata.X.copy()

    pcs = pd.DataFrame(-np.eye(10), index=test_adata.var_names)
    mean = pd.DataFrame(np.zeros((10,)), index=test_adata.var_names)
    print('pcs', pcs)
    print('mean', mean)

    X_proj = tc.project(test_adata, pcs, mean)

    np.testing.assert_allclose(X_proj, X_expected)

def test_chunk_proj(test_adata, tmpdir_factory):
    X_expected = -test_adata.X.copy()

    pcs = pd.DataFrame(-np.eye(10), index=test_adata.var_names)
    mean = pd.DataFrame(np.zeros((10,)), index=test_adata.var_names)
    print('pcs', pcs)
    print('mean', mean)

    tmpdir = str(tmpdir_factory.mktemp("test_proj"))
    input_file_name = os.path.join(tmpdir, "input.h5ad")
    ad.AnnData(test_adata.X).write(input_file_name)
    adata = sc.read_h5ad(input_file_name, backed='r')

    X_proj = tc.project(adata, pcs, mean, chunk_size=2)
    adata.file.close()

    np.testing.assert_allclose(X_proj, X_expected)

def test_file_notchunked_proj(test_adata, tmpdir_factory):
    X_expected = -test_adata.X.copy()

    pcs = pd.DataFrame(-np.eye(10), index=test_adata.var_names)
    mean = pd.DataFrame(np.zeros((10,)), index=test_adata.var_names)
    print('pcs', pcs)
    print('mean', mean)

    tmpdir = str(tmpdir_factory.mktemp("test_proj"))
    input_file_name = os.path.join(tmpdir, "input.h5ad")
    ad.AnnData(scp.sparse.csr_matrix(test_adata.X)).write(input_file_name)
    adata = sc.read_h5ad(input_file_name, backed='r')

    X_proj = tc.project(adata, pcs, mean)
    adata.file.close()

    np.testing.assert_allclose(X_proj, X_expected)
