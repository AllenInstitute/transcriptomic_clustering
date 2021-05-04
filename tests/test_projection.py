import pytest

import anndata as ad
import numpy as np
import transcriptomic_clustering as tc

@pytest.fixture
def test_adata():
    adata = ad.AnnData(np.random.rand(10,10))
    return adata


def test_simple_proj(test_adata):
    pcs = -np.eye(10)
    X = tc.project(test_adata, pcs)

    np.testing.assert_allclose(X, -test_adata.X)
