from math import exp
from typing import Optional, Union, Sequence, List
import logging

import scanpy as sc
import scipy as scp
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.utils import check_random_state

from .utils.memory import memory

logger = logging.getLogger(__name__)

Mask = Union[Sequence[int], slice, np.ndarray]
DEFAULT_NCOMPS = 50

def pca(
        adata: sc.AnnData,
        cell_select: Optional[Union[int, Mask]]=None,
        gene_mask: Mask=None,
        use_highly_variable: bool=False,
        svd_solver: str='randomized',
        n_comps: Optional[int]=None,
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
        (cannot be used with gene_mask)
    svd_solver:
        supports several methods:
            'auto':
                Selects based on data size (see scikit)
            'full':
                Does full SVD, and then selects top n_comps (not recommended for large data)
            'arpack':
                Uses ARPACK Fortran SVD solver
            'randomized':
                Uses Halko, 2009 randomized SVD algorithm (recommended for large data)
            'iterative':
                Iteratively constructs a PCA solution chunk by chunk, requires chunk_size arg
    n_comps:
        number of principal components to calculate
        defaults to min(n_obs, n_vars)
    random_state:
        sets random state for repeatability
        None, int, or np.random.RandomState
    chunk_size:
        if provided fits via IncrementalPCA(chunk_size), ignoring svd_solver.

    Returns
    -------
    components
        Dataframe of principal components (rows=genes, columns=components)
    explained_variance_ratio
        Ratio of explained variance.
    explained_variance
        Explained variance, equivalent to the eigenvalues of the
        covariance matrix.
    mean
        Dataframe containing the subtracted mean

    """
    # Handle defaults
    if use_highly_variable and 'highly_variable' not in adata.var:
        raise ValueError(
            'use_highly_variable is true but '
            'AnnData.var does not contain "highly_variable"'
        )

    random_state = check_random_state(random_state)

    # Generate cell_mask if needed
    logger.debug("Selecting Cell Mask")
    if isinstance(cell_select, int):
        # random sample
        if cell_select > adata.n_obs:
            cell_select = adata.n_obs
        cell_mask = random_state.choice(adata.n_obs, cell_select, replace=False)
        cell_mask.sort()
    elif cell_select is not None:
        cell_mask = cell_select
    elif cell_select is None:
        cell_mask = slice(None)
    oidx, _ = adata._normalize_indices((cell_mask, slice(None))) # handle cell mask like anndata would
    oidx_bool = np.zeros((adata.n_obs,), dtype=bool)
    oidx_bool[oidx] = True
    n_cells = len(oidx)
    
    # Mask adata
    logger.debug("Selecting Gene Mask")
    if (gene_mask is not None) and use_highly_variable:
        raise ValueError('Cannot use gene_mask and use_highly_variable together')
    elif use_highly_variable:
        gene_mask = adata.var['highly_variable']
    elif gene_mask is None:
        gene_mask = slice(None)
    _, vidx = adata._normalize_indices((slice(None), gene_mask)) # handle gene mask like anndata would
    vidx_bool = np.zeros((adata.n_vars,), dtype=bool)
    vidx_bool[vidx] = True
    n_genes = len(vidx)
    
    # select n_comps
    max_comps = min(n_cells, n_genes) - 1
    if not n_comps:
        n_comps = min(max_comps, DEFAULT_NCOMPS)
    elif n_comps > max_comps:
        logger.debug(
            f'n_comps {n_comps} > min(n_obs={n_cells}, n_genes={n_genes}) -1\n'
            f'Setting n_comps to {max_comps}'
        )
        n_comps = max_comps

    # Estimate memory
    # TODO: create method in adata subclass for estimating memory size of .X, 
    if not chunk_size:
        n_obs = n_cells
        n_vars = adata.n_vars
        if not adata.is_view:  # .X on view will try to load entire X into memory
            itemsize = adata.X.dtype.itemsize
        else:
            itemsize = np.dtype(np.float64).itemsize
        process_memory_estimate = (n_obs * n_vars) * itemsize / (1024 ** 3)
        output_memory_estimate = ((n_obs * n_comps) + (n_vars * n_comps) + (n_comps * 2)) * itemsize / (1024 ** 3)
        
        chunk_size = memory.estimate_chunk_size(
            adata,
            process_memory=process_memory_estimate,
            output_memory=output_memory_estimate,
        )
    logger.debug(f'Running PCA on n_obs {n_obs} and n_vars {n_vars}: {process_memory_estimate} GB')
    # Run PCA
    if chunk_size >= n_cells:
        _pca = PCA(n_components=n_comps, svd_solver=svd_solver, random_state=random_state)
        logger.debug(f'loading into memory')
        X = np.zeros((n_cells, n_genes))

        x_start = 0
        for chunk, start, end in adata.chunked_X(10000):  # should also be estimate memory
            if scp.sparse.issparse(chunk):
                chunk = chunk.toarray()
            chunk = chunk[oidx_bool[start:end], :]  # not sure why these indexing have to be separate...
            chunk = chunk[:, vidx_bool]
            if chunk.shape[0] > 0:
                x_end = x_start + chunk.shape[0]
                X[x_start:x_end] = chunk
                x_start = x_end

        logger.debug(f'performing fit')
        _pca.fit(X)
    else:
        if svd_solver != 'auto':
            logger.warning('Ignoring svd_solver, using IncrementalPCA')
        _pca = IncrementalPCA(n_components=n_comps, batch_size=chunk_size)
        if n_cells < adata.n_obs:
            adata = adata[oidx, :]

        for chunk, _, _ in adata.chunked_X(chunk_size):
            if scp.sparse.issparse(chunk):
                chunk = chunk.toarray()
            chunk = chunk[oidx_bool[start:end], :]
            chunk = chunk[:, vidx_bool]
            _pca.partial_fit(chunk)

    logging.debug(f'explained_variance_ratios: {_pca.explained_variance_ratio_}')
    return (
        pd.DataFrame(_pca.components_.T, index=adata.var_names[vidx]),
        _pca.explained_variance_ratio_,
        _pca.explained_variance_,
        pd.DataFrame(_pca.mean_, index=adata.var_names[vidx])
    )


def filter_known_components(
        principal_components: Union[pd.DataFrame, pd.Series],
        known_components: Union[pd.DataFrame, pd.Series],
        similarity_threshold: Optional[float] = 0.7):
    """
        Filters out principal components which correlate strongly with the known modes

        Parameters
        ----------
        principal_components: pincipal components from dimension reduction,
                        index is gene names, columns are principal components
        known_components: eigen vectors of gene expressions to filter out
                        index is gene names, columns are known modes
        similarity_threshold: threshold of correlation coefficients

        Returns
        -------
        mask of components to keep

    """
    if isinstance(principal_components, pd.Series):
        principal_components = principal_components.to_frame()
    if isinstance(known_components, pd.Series):
        known_components = known_components.to_frame()

    pcs_index_sorted = principal_components.sort_index()
    kns_index_sorted = known_components.sort_index()

    if not pcs_index_sorted.index.equals(kns_index_sorted.index):
        raise ValueError("The indices (genes) of the principal components and the known modes do not match")

    mat_pcs = pcs_index_sorted.to_numpy()
    mat_kns = kns_index_sorted.to_numpy()

    n_pcs = mat_pcs.shape[1]
    n_evs = mat_kns.shape[1]

    corr_pc_ev = np.corrcoef(mat_pcs, mat_kns, rowvar=False)[:n_pcs, -n_evs:]

    corr_pcs = np.amax(abs(corr_pc_ev), axis=1)

    rm_pcs_mask = corr_pcs > similarity_threshold

    return ~rm_pcs_mask


def filter_ev_ratios_zscore(explained_variance_ratios, threshold=2):
    """
    Filters principal components based on the z-scores of their explained variance ratios

    Parameters
    ----------
    explained_variance_ratios: np.ndarray of (explained_variance/sum(explained_variance))

    Returns
    -------
    mask of components to keep

    """
    z_scores = scp.stats.zscore(explained_variance_ratios)
    logging.debug(f'zscores: {z_scores}')
    return z_scores > threshold


def filter_explained_variances_elbow(explained_variances):
    """
    Filters out principal components by removing those whose explained variance
    are beyond the elbow of the explained variance curve

    To get elbow:
        Draw diagonal line from first point to last point (slopes down to left).
        Draw perpendicular lines from diagonal line to each point
        Select point with maximum distance to line, as long as it is below the line

    Parameters
    ----------
    explained_variances: array of explained variances

    Returns
    -------
    mask of components to keep

    """
    vars = np.asarray(explained_variances)
    vars.sort()
    vars = vars[::-1]

    dy = vars[-1] - vars[0]
    dx = len(vars) - 1
    l2 = np.sqrt(dy ** 2 + dx ** 2)
    dx /= l2
    dy /= l2

    dy0 = vars - vars[0]
    dx0 = np.arange(len(vars))

    parallel_l2 = np.sqrt((dx0 * dx) ** 2 + (dy0 * dy) ** 2)
    normal_x = dx0 - dx * parallel_l2
    normal_y = dy0 - dy * parallel_l2
    normal_l2 = np.sqrt(normal_x ** 2 + normal_y ** 2)

    #Picking the maximum normal that lies below the line.
    #If the entire curve is above the line, we just pick the last point.
    below_line = np.logical_and(normal_x < 0, normal_y < 0)
    remove_comps = np.zeros((len(explained_variances),), dtype=bool)
    if below_line.sum() > 0:
        max_index = np.argmax(normal_l2[below_line]) # index in below line
        max_index = np.flatnonzero(below_line)[max_index] # index in  vars
        remove_comps[max_index+1:] = 1
    
    return ~remove_comps



def filter_components(
        pcs: pd.DataFrame,
        explained_vars: np.ndarray,
        explained_var_ratios: np.ndarray, 
        known_components: Optional[pd.DataFrame]=None,
        similarity_threshold: float=0.7,
        method: Optional[str]='zscore',
        zth: Optional[float]=2,
        max_pcs: Optional[int]=None,
    ) -> pd.DataFrame:
    """"
    Function for filtering pca components

    Parameters
    ----------
    pcs: principal component dataframe, index=gene column=pcs
    explained_vars: array of explained variance of each component
    explained_var_ratios: array of explained variance ratios of each component
    known_components: components to remove (dataframe index=gene column=pcs) based on correlation
    similarity_threshold: correlation threshold for known components
    methods: 
        zscore:
            remove principal components whose explained variance ratios fall below the zth threshold
        elbow:
            remove principal components whose explained variances are below the elbow (dimenishing returns)
        None:
            no filtering
        
    zth: threshold for zscore filter
    max_pcs: maximum PCAs to keep

    Returns
    -------
    pd.Dataframe of down selected principal components

    """

    if known_components is not None:
        keep_pcs_mask = filter_known_components(
            pcs,
            known_components,
            similarity_threshold=similarity_threshold
        )
        pcs = pcs.iloc[:,keep_pcs_mask]
        explained_vars = explained_vars[keep_pcs_mask]
        explained_var_ratios = explained_var_ratios[keep_pcs_mask]

    if method == 'zscore':
        keep_pcs_mask = filter_ev_ratios_zscore(explained_var_ratios, threshold=zth)
    elif method == 'elbow':
        keep_pcs_mask = filter_explained_variances_elbow(explained_vars)
    elif method is None:
        keep_pcs_mask = np.zeros((pcs.shape[1],), dtype=bool)
    else:
        raise ValueError(f'principal component filter method {method} not recognized')

    pcs = pcs.iloc[:,keep_pcs_mask]
    explained_vars = explained_vars[keep_pcs_mask]
    explained_var_ratios = explained_var_ratios[keep_pcs_mask]
    
    if max_pcs is not None and max_pcs < pcs.shape[1]:
        pcs = pcs.iloc[:,:max_pcs]

    return pcs