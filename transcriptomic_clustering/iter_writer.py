import logging

import numpy as np
import scipy as scp
import anndata as ad
import h5py

logger = logging.getLogger(__name__)

class AnnDataIterWriter():
    """
    Class to handle iteratively writing filebacked AnnData Objects
    """
    def __init__(self, filename, initial_chunk, obs, var, dtype=None):
        self.issparse = scp.sparse.issparse(initial_chunk)
        self.initialize_file(filename, initial_chunk, obs, var, dtype=dtype)
        self.adata = ad.read_h5ad(filename, backed='r+')


    def initialize_file(self, filename, initial_chunk, obs, var, dtype=None):
        """Uses initial chunk to determine grouptype"""

        with h5py.File(filename, "w") as f:
            if self.issparse:
                if dtype is not None:
                    logger.warning("Ignoring dtype for sparse matrix")
                # TODO: change indextype based on nobs, nvars.
                initial_chunk.indptr = initial_chunk.indptr.astype(np.int64)
                initial_chunk.indices = initial_chunk.indices.astype(np.int64)
                ad._io.h5ad.write_csr(f, "X", initial_chunk)
            else:
                if dtype is None:
                    dtype = initial_chunk.dtype
                initial_chunk = np.atleast_2d(initial_chunk)
                ad._io.h5ad.write_array(
                    f, "X", initial_chunk,
                    dataset_kwargs={
                        'maxshape': (None, initial_chunk.shape[1]),
                        'dtype': dtype
                    }
                )
            ad._io.h5ad.write_dataframe(f, "obs", obs)
            ad._io.h5ad.write_dataframe(f, "var", var)


    def add_chunk(self, chunk):
        if self.issparse:
            self.adata.X.append(chunk)
        else:
            chunk = np.atleast_2d(chunk)
            chunk_nrows = chunk.shape[0]
            self.adata.X.resize(
                (self.adata.X.shape[0] + chunk_nrows),
                axis = 0
            )
            self.adata.X[-chunk_nrows:] = chunk
