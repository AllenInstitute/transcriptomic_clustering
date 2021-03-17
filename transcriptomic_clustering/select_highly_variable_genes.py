from typing import Optional, List, Literal

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from skmisc.loess import loess, loess_anova
from scipy.stats import rankdata, norm

from statsmodels.stats.multitest import fdrcorrection

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

def hicat_hvg(mtx: csr_matrix, genes: List[str], max_genes: Optional[int] = 3000):
    """
        select highly variable genes

        Parameters
        ----------
        mtx: transposed norm of cell expressions, rows correspond to genes and columns to cells
        genes: list of genes
        max_genes: max number of genes to be selected, default is 3000

        Returns
        -------
        highly variable genes: list of genes, with means and dispersions

    """

    mtx.data = 2**mtx.data -1

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
    positive_filter = (not np.isnan(dispersions).all()) and (dispersions > 0)

    means_filtered = means[positive_filter]
    dispersions_filtered = dispersions[positive_filter]
    genes_filtered = genes[positive_filter]

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
    
    df.sort_values(
        ['p_adj', 'z_score'],
        ascending=[True, False],
        na_position='last',
        inplace=True,
    )

    hvg_list = df['gene'][0:max_genes].tolist()

    hvg_dict = {}

    for i in genes:
        if i in hvg_list:
            hvg_dict[i] = True
        else:
            hvg_dict[i] = False

    hvg_ser = pd.Series(data=hvg_dict, index=hvg_dict.keys())

    df_hvg = pd.DataFrame(index=np.array(genes))
    df_hvg['means'] = means
    df_hvg['dispersions'] = dispersions
    df_hvg['highly_variable'] = hvg_ser
    
    return df_hvg

def select_hvg(ad_norm: ad.AnnData, 
            selected_cells: Optional[List], 
            low_thresh: Optional[int] = 1, 
            min_cells: Optional[int] = 4, 
            max_genes: Optional[int] = 3000,
            flavor: Literal['hicat', 'seurat', 'cell_ranger'] = 'hicat'):
    """
        select highly variable genes

        Parameters
        ----------
        ad_norm: normalization of cell expression in AnnData format (csr_matrix is supported)
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes

        selected_cells: interest cells specified
        low_thresh: lower threshold, uesed for pre-select hvg
        min_cells: mininum cells, used for pre-select hvg
        max_genes: number of genes to be selected, default is 3000
        flavor: highly variable genes method, default uses "hicat"

        Returns
        -------
        highly variable genes: added var['highly_variable'] to the the input AnnData

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

    ad_norm_hvg = adata_cellsampled[:, pre_selected_genes]

    # hvg
    if flavor == 'hicat':

        sampled_genes = ad_norm_hvg.var_names
        df_hvg = hicat_hvg(ad_norm_hvg.X.transpose(), sampled_genes, max_genes)

        ad_norm_hvg.uns['hvg'] = {'flavor', 'hicat'}
        ad_norm_hvg.var['highly_variable'] = df_hvg['highly_variable'].values
        ad_norm_hvg.var['means'] = df_hvg['means'].values
        ad_norm_hvg.var['dispersions'] = df_hvg['dispersions'].values
    
    elif flavor == 'seurat':
        sc.pp.highly_variable_genes(ad_norm_hvg, n_top_genes=max_genes)

    elif flavor == 'cell_ranger':
        sc.pp.highly_variable_genes(ad_norm_hvg, flavor='cell_ranger', n_top_genes=max_genes)

    else:
        raise ValueError('`flavor` needs to be "hicat", "seurat" or "cell_ranger"')

    return ad_norm_hvg

