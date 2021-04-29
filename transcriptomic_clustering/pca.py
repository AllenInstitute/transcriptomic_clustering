from typing import Optional, Union, Sequence
import logging

import scanpy as sc
import scipy as scp
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils import check_random_state

from .utils.memory import memory

Mask = Union[Sequence[int], slice, np.ndarray]

def pca(
        adata: sc.AnnData,
        cell_select: Optional[Union[int, Mask]]=None,
        gene_mask: Mask=None,
        use_highly_variable: bool=False,
        svd_solver: str='auto',
        n_comps: Optional[int]=None,
        zero_center: bool=True,
        random_state: Union[None, int, np.random.RandomState]=None,
        chunk_size: Optional[int]=None,
    ):
    """
    Runs PCA on Annotation Data. Uses scikit-learn PCA algorithms.

    Parameters:
    -----------
    adata: Annotated Data
        If AnnotatedData.X is sparse (e.g. scipy.csr_matrix), 
    cell_select:
        Based on type:
            :class:`int`
                Will select `cell_select` random cells without replacement for pca
            :term:`mask` (a list, tuple, slice, ndarray, etc)
                Will use `cell_select` cells for pca
    gene_mask:
        :term:`mask` (a list, tuple, slice, ndarray, etc)
            Will use 'gene_mask' cells for pca.
    use_highly_variable:
        Will only use highly variable genes 
        (if gene_select is also set, will only use highly variable genes in gene_select)
    svd_solver:
        supports several methods, with restrictions:
            'auto':
    n_comps:
        number of principle components to calculate
        defaults to min(n_obs, n_vars)
    random_state:
        sets random state for repeatability
        None, int, or np.random.RandomState
    chunk_size:
        if provided fits via IncrementalPCA(chunk_size), ignoring svd_solver.

    Returns
    -------
    components
        The principal components containing the loadings.
    explained_variance_ratio
        Ratio of explained variance.
    explained_variance
        Explained variance, equivalent to the eigenvalues of the
        covariance matrix.

    """
    # Handle defaults
    if use_highly_variable and  'highly_variable' not in adata.var:
        raise ValueError(
            'use_highly_variable is true but '
            'AnnData.var does not contain "highly_variable"'
        )

    random_state = check_random_state(random_state)

    # Generate cell_mask if needed
    if isinstance(cell_select, int):
        # random sample
        if cell_select > adata.n_obs:
            cell_select = adata.n_obs
        cell_mask = np.random.choice(adata.n_obs, cell_select, replace=False).sort()
    else:
        cell_mask = cell_select
    
    # Mask adata
    if cell_mask or gene_mask or use_highly_variable:
        cell_mask = slice(None) if not cell_mask else cell_mask
        gene_mask = slice(None) if not gene_mask else gene_mask
        adata_masked = adata[cell_mask, gene_mask]
        if use_highly_variable:
            adata_masked = adata_masked[:, adata_masked.var['highly_variable']]
    else:
        adata_masked = adata
    
    # select n_comps
    if not n_comps:
        n_comps = min(adata_masked.n_obs, adata_masked.n_vars, 51) - 1

    # Estimate memory
    # TODO: create method in adata subclass for estimating memory size of .X, 
    if not chunk_size:
        n_obs = adata_masked.n_obs
        n_vars = adata_masked.n_vars
        process_memory_estimate = (n_obs * n_vars) * 8 / (1024 ** 3)
        output_memory_estimate = ((n_obs * n_comps) + (n_vars * n_comps) + (n_comps * 2)) * 8 / (1024 ** 3)
        
        chunk_size = memory.estimate_chunk_size(
            adata_masked,
            process_memory=process_memory_estimate,
            output_memory=output_memory_estimate,
        )

    # Run PCA
    if chunk_size >= adata_masked.n_obs:
        _pca = PCA(n_components=n_comps, svd_solver=svd_solver, random_state=random_state)
        X = adata_masked.X
        if scp.sparse.isspmatrix(X):
            X = X.toarray()
        _pca.fit(X)
    else:
        if svd_solver != 'auto':
            logging.warning('Ignoring svd_solver, using IncrementalPCA')
        _pca = IncrementalPCA(n_components=n_comps, batch_size=chunk_size)
        for chunk, _, _ in adata_masked.chunked_X(chunk_size):
            _pca.partial_fit(chunk)

    return (
        _pca.components_,
        _pca.explained_variance_ratio_,
        _pca.explained_variance_,
    )