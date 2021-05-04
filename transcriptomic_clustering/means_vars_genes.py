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
        means: numpy array
        variances: numpy array
        gene_mask: boolean indicator of filtered genes

    """

    if adata.isbacked:
        # Estimate chunk size
        if not chunk_size:
            process_memory_est = adata.n_obs * adata.n_vars * (adata.X.dtype.itemsize / 1024 ** 2)
            chunk_size = tc.memory.estimate_chunk_size(
                                adata = adata,
                                process_memory=process_memory_est,
                                percent_allowed=50,
                                process_name='means_vars_genes'
                            )

        if chunk_size >= adata.n_obs:
            
            means, variances = sc.pp._utils._get_mean_var(np.expm1(adata.X[()]), axis=0)
            
            mtx_hg = (adata.X[()] > low_thresh).sum(axis=0) >= min_cells
            if hasattr(adata.X, "format_str"):
                if adata.X.format_str == "csr":
                    gene_mask = mtx_hg.tolist()[0]
                else:
                    gene_mask = mtx_hg.tolist()
            else:
                if issparse(adata.X):
                    gene_mask = mtx_hg.tolist()[0]
                else:
                    gene_mask = mtx_hg.tolist()

            return means[gene_mask], variances[gene_mask], gene_mask
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
                    
            mtx_hg = sum_hcpm >= min_cells
            gene_mask = mtx_hg.tolist()
            
            return w_mat.mean[gene_mask], w_mat.var_p[gene_mask], gene_mask
    else:
        means, variances = sc.pp._utils._get_mean_var(np.expm1(adata.X), axis=0)
            
        mtx_hg = (adata.X > low_thresh).sum(axis=0) >= min_cells
        if issparse(adata.X):
            gene_mask = mtx_hg.tolist()[0]
        else:
            gene_mask = mtx_hg.tolist()

        return means[gene_mask], variances[gene_mask], gene_mask
