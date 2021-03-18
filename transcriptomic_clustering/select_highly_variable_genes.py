from typing import Optional, List, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, csc_matrix
from skmisc.loess import loess
from scipy.stats import norm

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

def hicat_hvg(mtx: csc_matrix, genes: np.ndarray, max_genes: Optional[int] = 3000):
    """
        select highly variable genes

        Parameters
        ----------
        mtx: transposed norm of cell expressions
            rows correspond to genes and columns to cells
        genes: list of genes
        max_genes: number of highly variable genes to keep, default is 3000

        Returns
        -------
        dataframe of highly variable genes:
            df['highly_variable']: boolean indicator of highly variable genes
            df['means']: means per gene
            df['dispersions']: dispersions per gene

    """

    if mtx.getformat() == 'csc':
        mtx.data = 2**mtx.data -1
    else:
        raise ValueError("Unsupported format for cell_expression matrix. Must be in CSR format")

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

def select_highly_variable_genes(ad_norm: sc.AnnData,
            selected_cells: Optional[List] = None, 
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

        selected_cells: interested cells
        low_thresh: lower threshold, uesed for pre-select highly variable genes
        min_cells: mininum cells, used for pre-select highly variable genes
        max_genes: number of highly variable genes to keep
        flavor: highly variable genes method, default uses "hicat"

        Returns
        -------
        highly variable genes: boolean indicator of highly variable genes
                            var['highly_variable'] in AnnData

    """

    if not isinstance(ad_norm, sc.AnnData):
        raise ValueError('`select_highly_variable_genes` expects an `AnnData` argument, ')

    # sampling by cells
    # sampling by cells
    if selected_cells is not None:
        adata_cellsampled = ad_norm[selected_cells, :]
        
        # pre select hvg
        mtx_high = ad_norm.X > low_thresh
        arr_high = mtx_high.sum(axis=0) >= min_cells
        
        list_high = arr_high.tolist()[0]
        indices_high = [i for i, x in enumerate(list_high) if x == True]

        all_genes = adata_cellsampled.var_names
        pre_selected_genes = all_genes[indices_high].tolist()

        ad_norm_hvg = adata_cellsampled[:, pre_selected_genes]
        
    else:
        # pre select hvg
        mtx_high = ad_norm.X > low_thresh
        arr_high = mtx_high.sum(axis=0) >= min_cells
        
        list_high = arr_high.tolist()[0]
        indices_high = [i for i, x in enumerate(list_high) if x == True]

        all_genes = ad_norm.var_names
        pre_selected_genes = all_genes[indices_high].tolist()

        ad_norm_hvg = ad_norm[:, pre_selected_genes]

    # hvg
    if flavor == 'hicat':

        sampled_genes = ad_norm_hvg.var_names
        df_hvg = hicat_hvg(ad_norm_hvg.X.transpose(), sampled_genes.to_numpy(), max_genes)

        ad_norm_hvg.uns['hvg'] = {'flavor', 'hicat'}
        ad_norm_hvg.var['highly_variable'] = df_hvg['highly_variable']
        ad_norm_hvg.var['means'] = df_hvg['means'].values
        ad_norm_hvg.var['dispersions'] = df_hvg['dispersions'].values
    
    elif flavor == 'seurat':
        sc.pp.highly_variable_genes(ad_norm_hvg, n_top_genes=max_genes)

    elif flavor == 'cell_ranger':
        sc.pp.highly_variable_genes(ad_norm_hvg, flavor='cell_ranger', n_top_genes=max_genes)

    else:
        raise ValueError('`flavor` needs to be "hicat", "seurat" or "cell_ranger"')

    return ad_norm_hvg

