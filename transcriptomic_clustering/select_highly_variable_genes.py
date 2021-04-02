from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from skmisc.loess import loess
from scipy.stats import norm

from statsmodels.stats.multitest import fdrcorrection


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

def select_highly_variable_genes(adata: sc.AnnData,
            max_genes: Optional[int] = 3000,
            inplace: bool = True
            ) -> Optional[pd.DataFrame]:
    """
        select highly variable genes using the method in scrattch.hicat that
        is based on brennecke’s method, which assumes the reads follow a negative binomial distribution, 
        in which case, using loess fit to fine a relationship between mean and dispersions

        Parameters
        ----------
        adata: CPM normalization of cell expression (w/o logarithmized) in AnnData format (csr_matrix is supported)
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes
        max_genes: number of highly variable genes to keep
        inplace: whether to place calculated metrics in `.var` or return them.

        Returns
        -------
        Depending on `inplace` returns calculated metrics (:class:`pandas.DataFrame`) or
        updates `.var` with the following fields

        highly_variable: boolean indicator of highly-variable genes
        p_adj: p-adjust per gene
        z_score; z-score per gene
        means: means per gene
        dispersions: dispersions per gene

    """

    if not isinstance(adata, sc.AnnData):
        raise ValueError('`select_highly_variable_genes` expects an `AnnData` argument')

    if issparse(adata.X):
        if not isinstance(adata.X, csr_matrix):
            raise ValueError("Unsupported format for cell_expression matrix. Must be in CSR format")

    # means, variances
    means, variances = sc.pp._utils._get_mean_var(adata.X)

    #  dispersions
    dispersions = np.log10(variances / (means + 1e-10))

    # z-scores
    z_scores = compute_z_scores(dispersions)

    # Loess regression
    positive_filter = (not np.isnan(dispersions).all()) and (dispersions > 0)

    means_filtered = means[positive_filter]
    dispersions_filtered = dispersions[positive_filter]
    genes_filtered = adata.var_names[positive_filter]

    x = np.log10(means_filtered)
    y = dispersions_filtered

    loess_regression = loess(x, y)
    loess_regression.fit()

    loess_fit = loess_regression.outputs

    # p values
    loess_z = compute_z_scores(loess_fit.fitted_residuals)
    p_vals = 1 - norm.cdf(loess_z)
    
    # p.adjust using BH method
    p_adj = fdrcorrection(p_vals)[1]

    # select highly variable genes
    indices = [i for i, x in enumerate(p_adj) if x < 1]

    df = pd.DataFrame(index=indices)

    df['gene'] = genes_filtered[indices]
    df['p_adj'] = p_adj[indices]
    df['z_score'] = z_scores[indices]
    df['means'] = means_filtered[indices]
    df['dispersions'] = dispersions_filtered[indices]
    
    df.sort_values(
        ['p_adj', 'z_score'],
        ascending=[True, False],
        na_position='last',
        inplace=True,
    )

    df['highly_variable'] = False
    df.highly_variable.iloc[:max_genes] = True
    df = df[0:max_genes]

    if inplace:
        # filter by hvgs
        adata._inplace_subset_var(df.gene.tolist())

        adata.uns['hvg'] = {'flavor': 'hicat'}
        adata.var['highly_variable'] = df['highly_variable'].values
        adata.var['p_adj'] = df['p_adj'].values
        adata.var['z_score'] = df['z_score'].values
        adata.var['means'] = df['means'].values
        adata.var['dispersions'] = df['dispersions'].values
    else:
        return df

