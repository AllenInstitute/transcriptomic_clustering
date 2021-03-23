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

def hicat_hvg(adata: sc.AnnData, max_genes: Optional[int] = 3000, is_norm: bool = True):
    """
        select highly variable genes

        Parameters
        ----------
        adata: cell expression
        max_genes: number of highly variable genes to keep, default is 3000
        is_norm: whether input is normalized

        Returns
        -------
        AnnData with highly variable genes:
            .var['highly_variable']: boolean indicator of highly variable genes
            .var['means']: means per gene
            .var['dispersions']: dispersions per gene

    """

    if not isinstance(adata, sc.AnnData):
        raise ValueError("Unsupported format for cell_expression matrix, must be in AnnData format")

    if not isinstance(adata.X, csr_matrix):
        raise ValueError("Unsupported format for cell_expression matrix. Must be in CSR format")

    if is_norm:
        mtx = adata.X.transpose()
        mtx.data = 2**mtx.data -1
    else:
        sc.pp.normalize_total(adata, target_sum=1e6, inplace=True)
        mtx = adata.X.transpose()

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
    
    df.sort_values(
        ['p_adj', 'z_score'],
        ascending=[True, False],
        na_position='last',
        inplace=True,
    )

    hvg_list = df['gene'][0:max_genes].tolist()

    hvg_dict = {}

    for i in adata.var_names:
        if i in hvg_list:
            hvg_dict[i] = True
        else:
            hvg_dict[i] = False

    hvg_ser = pd.Series(data=hvg_dict, index=hvg_dict.keys())

    adata.uns['hvg'] = {'flavor': 'hicat'}
    adata.var['highly_variable'] = hvg_ser
    adata.var['means'] = means
    adata.var['dispersions'] = dispersions
    
    return adata

def select_highly_variable_genes(ad_norm: sc.AnnData,
            selected_cells: Optional[List] = None, 
            low_thresh: Optional[int] = 1, 
            min_cells: Optional[int] = 4, 
            max_genes: Optional[int] = 3000,
            flavor: Literal['hicat', 'seurat', 'cell_ranger'] = 'hicat',
            is_norm: bool = True):
    """
        select highly variable genes

        Parameters
        ----------
        ad_norm: normalization of cell expression in AnnData format (csr_matrix is supported)
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes

        selected_cells: interested cells
        low_thresh: lower threshold, uesed for pre-select highly variable genes
        min_cells: mininum cells, used for pre-select highly variable genes
        max_genes: number of highly variable genes to keep
        flavor: highly variable genes method, default uses "hicat"
        is_norm: check whether input the normalizaition of cell expression

        Returns
        -------
        highly variable genes: boolean indicator of highly variable genes
                            var['highly_variable'] in AnnData

    """

    if not isinstance(ad_norm, sc.AnnData):
        raise ValueError('`select_highly_variable_genes` expects an `AnnData` argument, ')

    # sampling by cells
    if selected_cells is not None:
        selected_cells = list(set(selected_cells))
        ad_norm.obs_names_make_unique()
        adata_cellsampled = ad_norm[selected_cells, :]
        
        # filtering genes
        mtx_high = (ad_norm.X > low_thresh).sum(axis=0) >= min_cells
        list_high = mtx_high.tolist()[0]
        indices_high = [i for i, x in enumerate(list_high) if x == True]
        
        pre_selected_genes = ad_norm.var_names[indices_high].tolist()
        pre_selected_genes = list(set(pre_selected_genes))

        adata_cellsampled.var_names_make_unique()
        ad_norm_hvg = adata_cellsampled[:, pre_selected_genes]
        
    else:
        # filtering genes
        mtx_high = (ad_norm.X > low_thresh).sum(axis=0) >= min_cells
        list_high = mtx_high.tolist()[0]
        indices_high = [i for i, x in enumerate(list_high) if x == True]
        
        pre_selected_genes = ad_norm.var_names[indices_high].tolist()
        pre_selected_genes = list(set(pre_selected_genes))
        
        ad_norm.var_names_make_unique()
        ad_norm_hvg = ad_norm[:, pre_selected_genes]

    # hvg
    if flavor == 'hicat':
        ad_norm_hvg = hicat_hvg(ad_norm_hvg, max_genes, is_norm)
    
    elif flavor == 'seurat':
        sc.pp.highly_variable_genes(ad_norm_hvg, n_top_genes=max_genes)

    elif flavor == 'cell_ranger':
        sc.pp.highly_variable_genes(ad_norm_hvg, flavor='cell_ranger', n_top_genes=max_genes)

    else:
        raise ValueError('`flavor` needs to be "hicat", "seurat" or "cell_ranger"')

    return ad_norm_hvg

