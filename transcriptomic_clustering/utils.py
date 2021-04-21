from typing import Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, vstack
from welford import Welford


def select_cells(adata: sc.AnnData,
            selected_cells: Optional[List] = None):
    """
        select cells

        Parameters
        ----------
        adata: cell expression in AnnData format (csr_matrix is preferred)
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes

        selected_cells: interested cells

        Returns
        -------
        adata with only seleceted cells

    """

    if not isinstance(adata, sc.AnnData):
        raise ValueError('`select_cells` expects an `AnnData` argument')

    # make cells' names unique
    adata.obs_names_make_unique()
    
    # sampling by interested cells
    if selected_cells is not None:
        selected_cells = list(set(selected_cells))
        adata._inplace_subset_obs(selected_cells)

def select_genes(adata: sc.AnnData,
            selected_genes: Optional[List] = None):
    """
        select genes

        Parameters
        ----------
        adata: cell expression in AnnData format (csr_matrix is preferred)
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes

        selected_genes: interested genes

        Returns
        -------
        adata with only seleceted genes

    """

    if not isinstance(adata, sc.AnnData):
        raise ValueError('`select_genes` expects an `AnnData` argument')

    # make genes' names unique
    adata.var_names_make_unique()
    
    # sampling by interested genes
    if selected_genes is not None:
        selected_genes = list(set(selected_genes))
        adata._inplace_subset_var(selected_genes)

def filter_genes_by_thresholds(adata: sc.AnnData,
            low_thresh: Optional[int] = 1,
            min_cells: Optional[int] = 4):
    """
        filter genes by thresholds

        Parameters
        ----------
        adata: log(CPM+1) normalization cell expression in AnnData format (csr_matrix is perferred)
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes

        low_thresh: lowest value required for a gene to pass filtering.
        min_cells: minimum number of cells expressed required for a gene to pass filtering.

        Returns
        -------
        adata with filtered genes

    """
    if not isinstance(adata, sc.AnnData):
        raise ValueError('`select_genes` expects an `AnnData` argument')

    mtx_high = (adata.X > low_thresh).sum(axis=0) >= min_cells

    if isinstance(adata.X, csr_matrix):
        list_high = mtx_high.tolist()[0]
        indices_high = [i_gene for i_gene, high_gene in enumerate(list_high) if high_gene == True]
    else:
        list_high = mtx_high.tolist()
        indices_high = [i_gene for i_gene, high_gene in enumerate(list_high) if high_gene == True]

    pre_selected_genes = adata.var_names[indices_high].tolist()
    pre_selected_genes = list(set(pre_selected_genes))

    adata.var_names_make_unique()
    adata._inplace_subset_var(pre_selected_genes)

def estimate_chunk_size(memory_required_to_run: Optional[int]):
    """
        esitmate chunk size

        TODO function, will be updated in the other issue
    """
    return 10000

def get_required_memory_in_GB(adata: sc.AnnData):
    """
        get required memory (GB)

        TODO function, will be updated in the other issue
    """
    return 5.0 # in GB


def get_gene_means_variances(adata: sc.AnnData, chunk_size: Optional[int] = None):
    """
        Calculate means and variances for each gene using Welford's online algorithm.

        Parameters
        ----------
        adata: normalization of cell expression in AnnData format
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes
        chunk_size: chunk size

        Returns
        -------
        means: numpy array
        variances: numpy array

    """
    memory_required_to_run = get_required_memory_in_GB(adata)

    if chunk_size is None:
        chunk_size = estimate_chunk_size(memory_required_to_run)

    if chunk_size >= adata.n_obs:
        return sc.pp._utils._get_mean_var(adata.X[()], axis=0)
    else:
        w_mat = Welford()

        for chunk, start, end in adata.chunked_X(chunk_size):

            if isinstance(chunk, csr_matrix):
                w_mat.add_all(np.expm1(chunk).toarray())
            else:
                w_mat.add_all(np.expm1(chunk))

        return w_mat.mean, w_mat.var_p