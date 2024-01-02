from typing import Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from welford import Welford
import warnings

import transcriptomic_clustering as tc

@usage_decorator
def get_means_vars_genes(adata: sc.AnnData,
                    low_thresh: Optional[int] = 1,
                    min_cells: Optional[int] = 4,
                    chunk_size: Optional[int] = None):
    """
        Calculate means and variances for each gene using Welford's online algorithm. 
        And filter genes by thresholds.

        Parameters
        ----------
        adata: normalization of cell expression in AnnData format
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes
        low_thresh: lowest value required for a gene to pass filtering.
        min_cells: minimum number of cells expressed required for a gene to pass filtering.
        chunk_size: chunk size

        Returns
        -------
        means: numpy array of mean expression for genes that pass threshold
        variances: numpy array of variances for genes that pass threshold
        gene_mask: boolean indicator of filtered genes

    """

    if adata.isbacked:
        return get_means_vars_genes_backed(adata, low_thresh, min_cells, chunk_size)
    else:
        if chunk_size:
            warnings.warn("In memory processing does not support chunking. "
                          "Ignoring `chunk_size` argument.")

        return get_means_vars_genes_inmemory(adata, low_thresh, min_cells)

@usage_decorator
def get_means_vars_genes_inmemory(
        adata: sc.AnnData,
        low_thresh: Optional[int] = 1,
        min_cells: Optional[int] = 4,

):
    """
    Get means, variances and gene mask when data is in memory
    See description of get_means_vars_genes() for details
    """

    means, variances = sc.pp._utils._get_mean_var(np.expm1(adata.X), axis=0)

    matrix_gene_mask = (adata.X > low_thresh).sum(axis=0) >= min_cells

    if issparse(adata.X):
        gene_mask = matrix_gene_mask.tolist()[0]
    else:
        gene_mask = matrix_gene_mask.tolist()

    return means[gene_mask], variances[gene_mask], gene_mask

@usage_decorator
def get_means_vars_genes_backed(
        adata: sc.AnnData,
        low_thresh: Optional[int] = 1,
        min_cells: Optional[int] = 4,
        chunk_size: Optional[int]=None
):
    # Estimate chunk size
    if not chunk_size:
        if not adata.is_view:  # .X on view will try to load entire X into memory
            itemsize = adata.X.dtype.itemsize
        else:
            itemsize = np.dtype(np.float64).itemsize
        process_memory_est = adata.n_obs * adata.n_vars * (itemsize / 1024 ** 3)
        chunk_size = tc.memory.estimate_chunk_size(
            adata=adata,
            process_memory=process_memory_est,
            percent_allowed=25,
            process_name='means_vars_genes'
        )

    w_mat = Welford()
    num_cells_above_thresh = np.zeros(adata.n_vars).transpose()

    for chunk, start, end in adata.chunked_X(chunk_size):

        num_cells_above_thresh_chunk = (chunk > low_thresh).sum(axis=0)
        num_cells_above_thresh += np.squeeze(np.asarray(num_cells_above_thresh_chunk))

        if issparse(chunk):
            chunk_em1 = chunk.expm1().toarray()
        else:
            chunk_em1 = np.expm1(chunk)

        w_mat.add_all(chunk_em1)

    matrix_gene_mask = num_cells_above_thresh >= min_cells
    gene_mask = matrix_gene_mask.tolist()

    return w_mat.mean[gene_mask], w_mat.var_p[gene_mask], gene_mask
