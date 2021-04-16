from typing import Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, vstack


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

def select_genes_by_chunking(input_cpm_file: str,
            selected_genes: Optional[List] = None,
            chunk_size: Optional[int] = 3000):
    """
        select genes by chunking

        Parameters
        ----------
        input_cpm_file: file name of the CPM normalization of cell expression in AnnData format (csr_matrix is perferred)
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes

        selected_genes: interested genes
        chunk_size: chunk size

        Returns
        -------
        adata with only seleceted genes

    """
    adata = sc.read_h5ad(input_cpm_file, backed='r')
    adata.var_names_make_unique()

    for chunk, start, end in adata.chunked_X(chunk_size):

        obs_chunk = adata.obs[start:end]
        adata_chunk = sc.AnnData(chunk, obs=obs_chunk, var=adata.var)
        adata_chunk._inplace_subset_var(selected_genes)

        if start == 0:
            var = adata_chunk.var
            obs = obs_chunk
            x_mat = adata_chunk.X
        else:
            obs = pd.concat([obs, obs_chunk])
            x_mat = vstack((x_mat, adata_chunk.X), format='csr')

        del adata_chunk

    adata.file.close()

    adata_genefiltered = sc.AnnData(X=x_mat, obs=obs, var=var)

    return adata_genefiltered

def filter_genes(adata: sc.AnnData,
            low_thresh: Optional[int] = 1,
            min_cells: Optional[int] = 4):
    """
        filter by genes

        Parameters
        ----------
        adata: CPM normalization cell expression in AnnData format (csr_matrix is perferred)
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

def filter_genes_by_chunking(input_cpm_file: str,
                            low_thresh: Optional[int] = 1,
                            min_cells: Optional[int] = 4,
                            chunk_size: Optional[int] = 3000):
    """
        filter genes by chunking

        Parameters
        ----------
        input_cpm_file: file name of the CPM normalization cell expression in AnnData format (csr_matrix is perferred)
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes

        low_thresh: lowest value required for a gene to pass filtering.
        min_cells: minimum number of cells expressed required for a gene to pass filtering.
        chunk_size: chunk size

        Returns
        -------
        filtered genes: list

    """
    adata = sc.read_h5ad(input_cpm_file, backed='r')

    # init sum of high cpm genes > low threshold
    sum_hcpm = np.zeros(adata.n_vars).transpose()

    # update sum_hcpm by chunking
    for chunk, start, end in adata.chunked_X(chunk_size):

        obs_chunk = adata.obs[start:end]
        adata_chunk = sc.AnnData(chunk, obs=obs_chunk, var=adata.var)

        sum_hcpm_chunk = (adata_chunk.X > low_thresh).sum(axis=0)
        sum_hcpm += np.squeeze(np.asarray(sum_hcpm_chunk))

        del adata_chunk

    adata.file.close()

    mtx_high = sum_hcpm >= min_cells
    list_high = mtx_high.tolist()
    indices_high = [i_gene for i_gene, high_gene in enumerate(list_high) if high_gene == True]

    pre_selected_genes = adata.var_names[indices_high].tolist()

    return list(set(pre_selected_genes))
