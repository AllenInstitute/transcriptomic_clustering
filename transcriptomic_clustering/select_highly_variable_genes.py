from typing import Optional, List

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from skmisc.loess import loess, loess_anova
from scipy.stats import rankdata, norm

from statsmodels.stats.multitest import fdrcorrection

import functools

def compute_z_scores(dispersion: np.array):
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

def select_hvg(ad_norm:ad.AnnData, selected_cells:Optional[List], 
                low_thresh: Optional[int] = 1, min_cells: Optional[int] = 4, max_genes: Optional[int] = 3000):
    """
        select highly variable genes using Brennecke's method

        Parameters
        ----------
        ad_norm: normalization of cell expression in AnnData format (csr_matrix is supported)
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes

        selected_cells: interest cells specified
        low_thresh: lower threshold, uesed for pre-select hvg
        min_cells: mininum cells, used for pre-select hvg
        max_genes: number of genes to be selected, default is 3000

        Returns
        -------
        highly variable genes: list of genes

    """

    # sampling by cells
    if not selected_cells:
        adata_cellsampled = ad_norm[selected_cells, :]

    # pre select hvg
    m_high = adata_cellsampled.X
    m_high.data = m_high.data > low_thresh

    mtx_high = m_high.sum(axis=0) >= min_cells
    l_high = mtx_high.tolist()[0]
    indices_high = [i for i, x in enumerate(l_high) if x == True]

    all_genes = adata_cellsampled.var_names
    pre_selected_genes = all_genes[indices_high].tolist()

    ad_norm_sampled = adata_cellsampled[:, pre_selected_genes]

    # prepare data
    mtx = ad_norm_sampled.X.transpose()
    mtx.data = 2**mtx.data-1

    # means
    means = np.squeeze(np.asarray(mtx.mean(axis=1)))

    # variances
    mtx_square = mtx.copy()
    mtx_square.data **= 2
    variances = np.squeeze(np.asarray(mtx_square.mean(axis=1))) - np.square(means)

    #  dispersions
    dispersions = np.log10(variances / (means + 1e-10))

    # z-scores
    z_scores = compute_z_scores(dispersions)

    # Loess regression
    not_const = variances > 0

    x = np.log10(means[not_const])
    y = dispersions[not_const]

    loess_regression = loess(x, y)
    loess_regression.fit()

    loess_fit = loess_regression.outputs

    # p values
    select = (not np.isnan(dispersions).all()) and (dispersions > 0)
    base = np.min(loess_fit.fitted_values)
    diff = dispersions - base
    diff[select] = loess_fit.fitted_residuals[select]

    loess_z = compute_z_scores(diff)

    p_vals = 1 - norm.cdf(loess_z)

    # p.adjust using BH method
    q_vals = fdrcorrection(p_vals)
    p_adj = q_vals[1]

    # select highly variable genes
    indices = [i for i, x in enumerate(p_adj) if x < 1]

    cur_genes = ad_norm_sampled.var_names
    select_genes_padj = cur_genes[indices].tolist()

    df = pd.DataFrame(index=indices)

    df['gene'] = cur_genes[indices]
    df['p_adj'] = p_adj[indices]
    df['z_score'] = z_scores[indices]

    df.sort_values(
        ['p_adj', 'z_score'],
        ascending=[True, False],
        na_position='last',
        inplace=True,
    )

    return df['gene'][0:max_genes].tolist()
