from typing import Optional, List

import scanpy as sc
from scipy.sparse import csr_matrix


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

def filter_genes(adata: sc.AnnData,
            low_thresh: Optional[int] = 1,
            min_cells: Optional[int] = 4):
    """
        filter by genes

        Parameters
        ----------
        adata: cell expression in AnnData format (csr_matrix is perferred)
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
        indices_high = [i for i, x in enumerate(list_high) if x == True]
    else:
        list_high = mtx_high.tolist()
        indices_high = [i for i, x in enumerate(list_high) if x == True]

    pre_selected_genes = adata.var_names[indices_high].tolist()
    pre_selected_genes = list(set(pre_selected_genes))

    adata.var_names_make_unique()
    adata._inplace_subset_var(pre_selected_genes)
    