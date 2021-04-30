from typing import Optional, Union, Sequence
import scanpy as sc
import numpy as np
import warnings

from .utils.memory import memory

Mask = Union[Sequence[int], slice, np.ndarray]

def pca(adata: sc.AnnData, cell_select: Optional[Union[int, Mask]]=None, gene_mask: Mask=None, **kwargs):
    """
    Runs PCA on Annotation Data. Uses scanpy.tl.pca under the hood

    Parameters:
    -----------
    adata: Annotated Data
    cell_select:
        Based on type:
            :class:`int`
                Will select `cell_select` random cells without replacement for pca
            :term:`mask` (a list, tuple, slice, ndarray, etc)
                Will use `cell_select` cells for pca
    gene_mask:
        :term:`mask` (a list, tuple, slice, ndarray, etc)
            Will use 'gene_select' cells for pca.
    use_highly_variable:
        Will only use highly variable genes 
        (if gene_select is also set, will only use highly variable genes in gene_select)
    copy:
        If cell_select or gene_select is provided, this function will always return a copy

    Other kwargs:
        Pass all kwargs found in :func:`~scanpy.tl.pca`
        n_comps, zero_center, svd_solver, random_state, return_info, use_highly_variable, dtype, copy, chunked, chunk_size


    Returns
    -------
    Based on:
        `cell_select` or `gene_mask` or `copy`:
            returns an AnnData object with results in:
                `.obsm['X_pca']`
                PCA representation of data.
                `.varm['PCs']`
                    The principal components containing the loadings.
                `.uns['pca']['variance_ratio']`
                    Ratio of explained variance.
                `.uns['pca']['variance']`
                    Explained variance, equivalent to the eigenvalues of the
                    covariance matrix.
        Otherwise
            appends the above fields to the input AnnData object and returns None


    """
    # Ignore copy kwarg
    is_subset = False
    if (cell_select or gene_mask):
        is_subset = True
        kwargs['copy'] = False

    # Generate cell_mask if needed
    if isinstance(cell_select, int):
        # random sample
        if cell_select > adata.n_obs:
            cell_select = adata.n_obs
        cell_mask = np.random.choice(adata.n_obs, cell_select, replace=False).sort()
    else:
        cell_mask = cell_select
    
    # Mask adata
    if not is_subset:
        adata_masked = adata
    else:
        cell_mask = slice(None) if not cell_mask else cell_mask
        gene_mask = slice(None) if not gene_mask else gene_mask
        adata_masked = adata[cell_mask, gene_mask]
    
    # Estimate memory
    # TODO: create method in adata subclass for estimating memory size of .X, 
    # TODO: make scanpy/scikit not return x_pca, just pcs?
    n_obs = adata_masked.n_obs
    n_vars = adata_masked.n_vars
    n_comps = kwargs.get('n_comps', max(sc.settings.N_PCS, min(n_obs, n_vars)-1))
    process_memory_estimate = (n_obs * n_vars) * 8 / (1024 ** 3)
    output_memory_estimate = ((n_obs * n_comps) + (n_vars * n_comps) + (n_comps * 2)) * 8 / (1024 ** 3)
    if kwargs.get('copy') or is_subset:
        output_memory_estimate += process_memory_estimate
    
    n_chunks = memory.estimate_n_chunks(
        process_memory=process_memory_estimate,
        output_memory=output_memory_estimate,
    )

    # Run PCA
    if n_chunks == 1:
        output = sc.tl.pca(adata_masked, **kwargs)
    else:
        kwargs['chunk_size'] = memory.get_chunk_size(adata_masked, n_chunks)
        kwargs['chunked'] = True
        output = sc.tl.pca(adata_masked, **kwargs)

    # Handle subset case
    if is_subset is True:
        # data gets written to adata_masked, so we need to return it
        return adata_masked
    else:
        # Otherwise it's written directly to the adata object
        return output
