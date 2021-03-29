from typing import Optional, List, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, csc_matrix
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

def select_highly_variable_genes(ad_norm: sc.AnnData,
            selected_cells: Optional[List] = None,
            low_thresh: Optional[int] = 1,
            min_cells: Optional[int] = 4,
            max_genes: Optional[int] = 3000,
            in_place: bool = True,
            is_norm: bool = True
            ) -> Optional[pd.DataFrame]:
    """
        select highly variable genes using the method in scrattch.hicat

        Parameters
        ----------
        ad_norm: normalization of cell expression in AnnData format (csr_matrix is supported)
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes

        selected_cells: interested cells
        low_thresh: lower threshold, uesed for pre-select highly variable genes
        min_cells: mininum cells, used for pre-select highly variable genes
        max_genes: number of highly variable genes to keep
        in_place: if True it will update the input AnnData varialbe
        is_norm: check whether input the normalizaition of cell expression

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

    if not isinstance(ad_norm, sc.AnnData):
        raise ValueError('`select_highly_variable_genes` expects an `AnnData` argument')

    if not isinstance(ad_norm.X, csr_matrix):
        raise ValueError("Unsupported format for cell_expression matrix. Must be in CSR format")

    # sampling by cells
    if selected_cells is not None:
        selected_cells = list(set(selected_cells))
        ad_norm.obs_names_make_unique()
        ad_norm._inplace_subset_obs(selected_cells)
 
    # filtering genes
    mtx_high = (ad_norm.X > low_thresh).sum(axis=0) >= min_cells
    list_high = mtx_high.tolist()[0]
    indices_high = [i for i, x in enumerate(list_high) if x == True]

    pre_selected_genes = ad_norm.var_names[indices_high].tolist()
    pre_selected_genes = list(set(pre_selected_genes))

    ad_norm.var_names_make_unique()
    ad_norm._inplace_subset_var(pre_selected_genes)

    #
    if is_norm:
        mtx = ad_norm.X.transpose()
        mtx.data = 2**mtx.data -1
    else:
        sc.pp.normalize_total(ad_norm, target_sum=1e6, inplace=True)
        mtx = ad_norm.X.transpose()

    # means
    means = np.squeeze(np.asarray(mtx.mean(axis=1)))

    # variances
    mtx.data **= 2
    variances = np.squeeze(np.asarray(mtx.mean(axis=1))) - np.square(means)

    #  dispersions
    dispersions = np.log10(variances / (means + 1e-10))

    # z-scores
    z_scores = compute_z_scores(dispersions)

    # Loess regression
    positive_filter = (not np.isnan(dispersions).all()) and (dispersions > 0)

    means_filtered = means[positive_filter]
    dispersions_filtered = dispersions[positive_filter]
    genes_filtered = ad_norm.var_names[positive_filter]

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

    if in_place:
        # filtered by hvgs
        ad_norm._inplace_subset_var(df.gene.tolist())

        ad_norm.uns['hvg'] = {'flavor': 'hicat'}
        ad_norm.var['highly_variable'] = df['highly_variable'].values
        ad_norm.var['p_adj'] = df['p_adj'].values
        ad_norm.var['z_score'] = df['z_score'].values
        ad_norm.var['means'] = df['means'].values
        ad_norm.var['dispersions'] = df['dispersions'].values
    else:
        return df

