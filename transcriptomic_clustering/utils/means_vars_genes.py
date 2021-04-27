from typing import Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from welford import Welford

import transcriptomic_clustering as tc


def means_vars_genes(adata: sc.AnnData,
                    low_thresh: Optional[int] = 1,
                    min_cells: Optional[int] = 4,
                    chunk_size: Optional[int] = None,
                    process_memory: Optional[float]=None):
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
        process_memory: amount of memory in GB function is expected to need to process entire data

        Returns
        -------
        means: numpy array
        variances: numpy array
        filtered genes: list of genes

    """

    if chunk_size is None:
        memory = tc.utils.memory.Memory()
        if not process_memory:
            raise ValueError("please input either chunk_size or process_memory to run means_vars_genes")
        chunk_size = memory.estimate_chunk_size(adata, process_memory, percent_allowed = 50)

    if chunk_size >= adata.n_obs:
        
        means, variances = sc.pp._utils._get_mean_var(np.expm1(adata.X[()]), axis=0)
        
        mtx_high = (adata.X[()] > low_thresh).sum(axis=0) >= min_cells
        if hasattr(adata.X, "format_str"):
            if adata.X.format_str == "csr":
                list_high = mtx_high.tolist()[0]
            else:
                list_high = mtx_high.tolist()
        else:
            if issparse(adata.X):
                list_high = mtx_high.tolist()[0]
            else:
                list_high = mtx_high.tolist()
            
        indices_high = [i_gene for i_gene, high_gene in enumerate(list_high) if high_gene == True]
        pre_selected_genes = adata.var_names[indices_high].tolist()

        return means, variances, list(set(pre_selected_genes))
    else:
        w_mat = Welford()
        sum_hcpm = np.zeros(adata.n_vars).transpose()

        for chunk, start, end in adata.chunked_X(chunk_size):
            
            sum_hcpm_chunk = (chunk > low_thresh).sum(axis=0)
            sum_hcpm += np.squeeze(np.asarray(sum_hcpm_chunk))

            if issparse(chunk):
                if isinstance(chunk, csr_matrix):
                    w_mat.add_all(np.expm1(chunk).toarray())
                else:
                    raise ValueError("Unsupported format for cell_expression matrix. Must be in CSR or dense format")
            else:
                w_mat.add_all(np.expm1(chunk))
                
        mtx_high = sum_hcpm >= min_cells
        list_high = mtx_high.tolist()
        indices_high = [i_gene for i_gene, high_gene in enumerate(list_high) if high_gene == True]

        pre_selected_genes = adata.var_names[indices_high].tolist()

        return w_mat.mean, w_mat.var_p, list(set(pre_selected_genes))