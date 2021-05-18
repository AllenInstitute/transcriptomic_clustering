from typing import Optional, Union, Sequence

import numpy as np
import scipy as scp
import pandas as pd
import scanpy as sc
import anndata as ad
import transcriptomic_clustering as tc

Mask = Union[Sequence[int], slice, np.ndarray]

def project(
        adata: ad.AnnData,
        principal_comps: pd.DataFrame,
        mean: Optional[pd.DataFrame]=None,
        chunk_size: Optional[int]=None) -> np.ndarray:
    """
    Projects data into principal component space

    Parameters
    ----------
    adata:
        adata to project into principal component space
    principal_comps: 
        principal component Dataframe (rows=genes, columns=components)
    mean:
        mean used for zero centering (rows=genes, column=mean)

    Returns
    -------
    Adata object in principal component space
    """

    if not mean.index.equals(principal_comps.index):
        raise ValueError('mean and principal comps have different genes')
    _, vidx = adata._normalize_indices((slice(None), principal_comps.index)) # handle gene mask like anndata would
    principal_comps = principal_comps.to_numpy()
    mean = mean.to_numpy().T

    n_obs = adata.n_obs
    n_comps = principal_comps.shape[0]
    n_genes = principal_comps.shape[1]

    issparse = False
    if adata.isbacked and hasattr(adata.X, "format_str") and adata.X.format_str == "csr":
        issparse = True
    
    # Estimate memory
    if not chunk_size:
        itemsize = adata.X.dtype.itemsize
        process_memory = n_obs * n_genes * itemsize / (1024 ** 3)
        if issparse:
            process_memory *= 2

        output_memory = n_obs * n_comps * itemsize / (1024 ** 3)
        chunk_size = tc.memory.estimate_chunk_size(
            adata,
            process_memory=process_memory,
            output_memory=output_memory,
            percent_allowed=70,
            process_name='project',
        )

    # Transform
    pcs_T = principal_comps.T
    if not adata.isbacked and chunk_size >= n_obs:
        X = adata.X
        if issparse:
            X = X.toarray()
        X = X[:, vidx]
        if mean is not None:
            X -= mean
        X_proj = X @ pcs_T

    else:
        X_proj = np.zeros((adata.n_obs, n_comps))
        for chunk, start, end in adata.chunked_X(chunk_size):
            if scp.sparse.issparse(chunk):
                chunk = chunk.toarray()
            chunk = chunk[:, vidx]
            if mean is not None:
                chunk -= mean
            X_proj[start:end,:] = chunk @ pcs_T

    return X_proj