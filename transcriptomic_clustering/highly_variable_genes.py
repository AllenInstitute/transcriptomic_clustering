from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from skmisc.loess import loess
from scipy.stats import norm

from statsmodels.stats.multitest import fdrcorrection

@usage_decorator
def compute_z_scores(dispersion: np.ndarray):
    """
        Compute dispersion z-scores for each gene in a gene x sample matrix

        Parameters
        ----------
        dispersion: numpy array

        Returns
        -------
        z-scores: numpy array

    """
    q75, q25 = np.percentile(dispersion, [75 ,25])
    iqr = q75 - q25
    m_iqr = (q25 + q75)/2.0
    delta = iqr / (norm.ppf(0.75) - norm.ppf(0.25))
    
    return (dispersion  - m_iqr) / delta

@usage_decorator
def highly_variable_genes(adata: sc.AnnData,
            means: np.array,
            variances: np.array,
            gene_mask: list,
            max_genes: Optional[int] = 3000,
            annotate: bool = False
            ) -> Optional[pd.DataFrame]:
    """
        select highly variable genes using the method in scrattch.hicat that
        is based on brennecke’s method, which assumes the reads follow a negative binomial distribution, 
        in which case, using loess fit to find a relationship between means and dispersions

        Parameters
        ----------
        adata: log(CPM+1) normalization of cell expression in AnnData format
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes
        max_genes: number of highly variable genes to keep
        means: means of CPM normalization of cell expression
        variances: variances of CPM normalization of cell expression
        gene_mask: boolean indicator of filtered genes
        annotate: whether to place calculated metrics in `.var` or return them.

        Returns
        -------
        Depending on `annotate` returns calculated metrics (:class:`pandas.DataFrame`) or
        updates `.var` with the following fields

        highly_variable: boolean indicator of highly-variable genes
    """
    
    # dispersions
    dispersions = np.log1p(variances / (means + 1e-10))

    # z-scores
    z_scores = compute_z_scores(dispersions)

    # Loess regression
    x = np.log1p(means)
    y = dispersions

    loess_regression = loess(x, y)
    loess_regression.fit()

    loess_fit = loess_regression.outputs

    # p values
    loess_z = compute_z_scores(loess_fit.fitted_residuals)
    p_vals = 1 - norm.cdf(loess_z)
    
    # p.adjust using BH method
    rejected,p_adj = fdrcorrection(p_vals)

    # select highly variable genes
    qval_indices = [i_gene for i_gene, padj_val in enumerate(p_adj) if padj_val < 1]

    df = pd.DataFrame(index=qval_indices)

    select_genes = adata.var_names[gene_mask]

    df['gene'] = select_genes[qval_indices]
    df['p_adj'] = p_adj[qval_indices]
    df['z_score'] = z_scores[qval_indices]
    
    df.sort_values(
        ['p_adj', 'z_score'],
        ascending=[True, False],
        na_position='last',
        inplace=True,
    )

    hvg_set = set(df['gene'][0:max_genes].tolist())
    hvg_dict = {gene: (gene in hvg_set) for gene in adata.var_names}
    df = pd.Series(data=hvg_dict)

    if annotate:
        adata.uns['hvg'] = {'flavor': 'hicat'}
        adata.var['highly_variable'] = df
    
    return df

