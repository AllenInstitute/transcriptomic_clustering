import pytest
from unittest.mock import patch, MagicMock
import tempfile
import os

import numpy as np
import pandas as pd
import scipy as scp
from numpy.testing import assert_allclose

from transcriptomic_clustering.iter_writer import AnnDataIterWriter


def test_anndata_iter_writer_sparse():
    with tempfile.TemporaryDirectory() as tmp_dir:
        X = scp.sparse.csr_matrix(np.asarray([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]))
        adata_writer = AnnDataIterWriter(
            os.path.join(tmp_dir, 'test_adata'),
            X[0,:],
            pd.DataFrame(index=['obs1', 'obs2', 'obs3']),
            pd.DataFrame(index=['var1', 'var2', 'var3']),
        )
        adata_writer.add_chunk(X[1:3,:])
        for chunk, _, _ in adata_writer.adata.chunked_X(1):
            print(chunk)
        adata = adata_writer.adata.to_memory()
        assert_allclose(X.todense(), adata.X.todense())


def test_anndata_iter_writer_dense():
    with tempfile.TemporaryDirectory() as tmp_dir:
        X = np.asarray([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
        adata_writer = AnnDataIterWriter(
            os.path.join(tmp_dir, 'test_adata'),
            X[0,:],
            pd.DataFrame(index=['obs1', 'obs2', 'obs3']),
            pd.DataFrame(index=['var1', 'var2', 'var3']),
        )
        adata_writer.add_chunk(X[1,:])
        adata_writer.add_chunk(X[2,:])
        adata = adata_writer.adata.to_memory()
        assert_allclose(X, adata.X)