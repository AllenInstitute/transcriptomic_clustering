from typing import Optional, Union, Sequence

import numpy as np
import scipy as scp
import scanpy as sc
import anndata as ad
import transcriptomic_clustering as tc

Mask = Union[Sequence[int], slice, np.ndarray]

def project(
        adata: ad.AnnData,
        principle_comps: np.ndarray,
        mean: Optional[np.ndarray]=None,
        gene_mask: Optional[Mask]=None,
        use_highly_variable: bool=False,
        chunk_size: Optional[int]=None) -> np.ndarray:
    """
    Projects data into principle component space

    Parameters
    ----------
    adata:
        adata to project into principle component space
    principle_comps: 
        principle component matrix (n_comps x n_genes)
    mean:
        mean used for zero centering (pca output)

    Returns
    -------
    Adata object in principle component space
    """

    if (gene_mask is not None) and use_highly_variable:
            raise ValueError('Cannot use gene_mask and use_highly_variable together')
    elif use_highly_variable:
        gene_mask = adata.var['highly_variable']
    elif gene_mask is None:
        gene_mask = slice(None)
    _, vidx = adata._normalize_indices((slice(None), gene_mask)) # handle gene mask like anndata would
    
    n_obs = adata.n_obs
    n_comps = principle_comps.shape[0]
    n_genes = principle_comps.shape[1]

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
    pcs_T = principle_comps.T
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