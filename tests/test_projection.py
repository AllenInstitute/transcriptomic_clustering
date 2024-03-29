import os
import pytest

import numpy as np
import scipy as scp
import pandas as pd
import scanpy as sc
import anndata as ad
import transcriptomic_clustering as tc


@pytest.fixture
def adata():
    rng = np.random.default_rng(1)
    n_obs = 20
    n_vars = 10
    obs = pd.DataFrame(index=[f'cell_{i}' for i in range(n_obs)])
    var = pd.DataFrame(index=[f'gene_{i}' for i in range(n_vars)])

    adata = ad.AnnData(rng.random((20,10)), obs=obs, var=var)
    return adata

@pytest.fixture
def pcs(adata):
    return pd.DataFrame(-np.eye(10,5), index=adata.var_names, columns=[f'PC{i}' for i in range(5)])

@pytest.fixture
def mean(adata):
    return pd.DataFrame(np.zeros((10,)), index=adata.var_names)

@pytest.fixture
def x_exp(adata):
    return -adata.X[:,0:5].copy()


def test_simple_proj(adata, mean, pcs, x_exp):
    ad_proj = tc.project(adata, pcs, mean)

    np.testing.assert_allclose(ad_proj.X, x_exp)
    assert ad_proj.obs.index.equals(adata.obs.index)

def test_chunk_proj(adata, mean, pcs, x_exp, tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("test_proj"))
    input_file_name = os.path.join(tmpdir, "input.h5ad")
    ad.AnnData(adata.X, obs=adata.obs, var=adata.var).write(input_file_name)
    adata = sc.read_h5ad(input_file_name, backed='r')

    ad_proj = tc.project(adata, pcs, mean, chunk_size=2)
    adata.file.close()

    np.testing.assert_allclose(ad_proj.X, x_exp)

def test_file_notchunked_proj(adata, mean, pcs, x_exp, tmpdir_factory):
    tmpdir = str(tmpdir_factory.mktemp("test_proj"))
    input_file_name = os.path.join(tmpdir, "input.h5ad")
    ad.AnnData(scp.sparse.csr_matrix(adata.X), obs=adata.obs, var=adata.var).write(input_file_name)
    adata = sc.read_h5ad(input_file_name, backed='r')

    ad_proj = tc.project(adata, pcs, mean)
    adata.file.close()

    np.testing.assert_allclose(ad_proj.X, x_exp)
