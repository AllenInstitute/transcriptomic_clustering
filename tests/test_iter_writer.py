import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal
from scipy import sparse

from transcriptomic_clustering.iter_writer import AnnDataIterWriter


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("dtype", [None, np.float32, np.int16])
def test_anndata_iter_writer_roundtrip(tmp_path, as_sparse, dtype):
    values = np.arange(21, dtype=np.float64).reshape(7, 3)
    values[2:4] = 0  # Empty CSR rows, including a chunk boundary.
    matrix = sparse.csr_matrix(values) if as_sparse else values
    obs = pd.DataFrame(
        {"batch": pd.Categorical(["a", "b", "a", "b", "a", "b", "a"]),
         "count": np.arange(7, dtype=np.int32)},
        index=pd.Index([f"cell_{i}" for i in range(7)], name="cell"),
    )
    var = pd.DataFrame(
        {"selected": [True, False, True]},
        index=pd.Index(["gene_a", "gene_b", "gene_c"], name="gene"),
    )
    path = tmp_path / "roundtrip.h5ad"
    writer = AnnDataIterWriter(path, matrix[:3], obs, var, dtype=dtype)
    try:
        writer.add_chunk(matrix[3:6])
        writer.add_chunk(matrix[6:])  # Final short chunk.
        writer.add_chunk(matrix[:0])
    finally:
        writer.adata.file.close()

    for backed in [None, "r"]:
        result = ad.read_h5ad(path, backed=backed)
        try:
            actual = result.X[:]
            assert actual.dtype == (dtype or values.dtype)
            assert_array_equal(actual.toarray() if as_sparse else actual, values)
            assert_frame_equal(result.obs, obs)
            assert_frame_equal(result.var, var)
            assert_array_equal(result[3:7, :].X.toarray() if as_sparse
                               else result[3:7, :].X, values[3:7])
        finally:
            if result.isbacked:
                result.file.close()
    with h5py.File(path) as handle:
        assert handle.attrs["encoding-type"] == "anndata"
        assert handle["X"].attrs["encoding-type"] == ("csr_matrix" if as_sparse else "array")
        if as_sparse:
            assert handle["X/indptr"].dtype == np.int64
            assert handle["X/indices"].dtype == np.int64
            assert matrix.dtype == np.float64  # Writing must not mutate input.
        else:
            assert handle["X"].maxshape == (None, 3)


def test_anndata_iter_writer_dense_row_and_invalid_chunks(tmp_path):
    writer = AnnDataIterWriter(
        tmp_path / "rows.h5ad", np.array([1., 2.]),
        pd.DataFrame(index=["a", "b"]), pd.DataFrame(index=["x", "y"]),
    )
    try:
        with pytest.raises(ValueError, match="shape"):
            writer.add_chunk(np.zeros((1, 3)))
        with pytest.raises(ValueError, match="storage"):
            writer.add_chunk(sparse.csr_matrix([[3., 4.]]))
        writer.add_chunk(np.array([3., 4.]))
        with pytest.raises(ValueError, match="shape"):
            writer.add_chunk(np.array([5., 6.]))
        assert_array_equal(writer.adata.X[:], [[1., 2.], [3., 4.]])
    finally:
        writer.adata.file.close()
