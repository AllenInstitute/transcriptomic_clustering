import logging

import numpy as np
import scipy as scp
import anndata as ad
import h5py

logger = logging.getLogger(__name__)

class AnnDataIterWriter():
    """
    Append dense or CSR row chunks to an h5ad file.

    ``obs`` describes all rows, in chunk order; ``var`` describes all columns.
    The caller owns ``adata`` and must close ``adata.file`` after the last write.
    An explicit dtype applies to every chunk, including sparse data.
    """
    def __init__(self, filename, initial_chunk, obs, var, dtype=None):
        self.issparse = scp.sparse.issparse(initial_chunk)
        self.initialize_file(filename, initial_chunk, obs, var, dtype=dtype)
        self.adata = ad.read_h5ad(filename, backed='r+')


    def initialize_file(self, filename, initial_chunk, obs, var, dtype=None):
        """Write metadata with AnnData and extensible X using the h5ad encoding."""
        if self.issparse:
            initial_chunk = initial_chunk.tocsr(copy=True)
            if dtype is not None:
                initial_chunk = initial_chunk.astype(dtype)
        else:
            initial_chunk = np.atleast_2d(initial_chunk).astype(
                dtype or initial_chunk.dtype, copy=False
            )
        if initial_chunk.shape[1] != len(var) or initial_chunk.shape[0] > len(obs):
            raise ValueError("Chunk shape does not match obs/var")

        # Public serialization preserves categorical columns and index metadata.
        ad.AnnData(obs=obs.copy(), var=var.copy()).write_h5ad(filename)
        with h5py.File(filename, "a") as f:
            if self.issparse:
                group = f.create_group("X")
                group.attrs.update({
                    "encoding-type": "csr_matrix",
                    "encoding-version": "0.1.0",
                    "shape": initial_chunk.shape,
                })
                for name, values in (
                    ("data", initial_chunk.data),
                    ("indices", initial_chunk.indices.astype(np.int64)),
                    ("indptr", initial_chunk.indptr.astype(np.int64)),
                ):
                    group.create_dataset(name, data=values, maxshape=(None,))
            else:
                dataset = f.create_dataset(
                    "X", data=initial_chunk,
                    maxshape=(None, initial_chunk.shape[1]),
                )
                dataset.attrs.update({
                    "encoding-type": "array", "encoding-version": "0.2.0"
                })


    def add_chunk(self, chunk):
        if scp.sparse.issparse(chunk) != self.issparse:
            raise ValueError("All chunks must use the same dense/sparse storage")
        if self.issparse:
            chunk = chunk.tocsr().astype(self.adata.X.dtype, copy=False)
        else:
            chunk = np.atleast_2d(chunk)
        if (chunk.shape[1] != self.adata.n_vars or
                self.adata.X.shape[0] + chunk.shape[0] > self.adata.n_obs):
            raise ValueError("Chunk shape does not match remaining obs/var")
        if chunk.shape[0] == 0:
            return
        if self.issparse:
            self.adata.X.append(chunk)
        else:
            chunk_nrows = chunk.shape[0]
            self.adata.X.resize(
                (self.adata.X.shape[0] + chunk_nrows),
                axis = 0
            )
            self.adata.X[-chunk_nrows:] = chunk
