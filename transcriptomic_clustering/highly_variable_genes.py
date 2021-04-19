from typing import Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse
from skmisc.loess import loess
from scipy.stats import norm
from welford import Welford
from anndata import AnnData

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

def select_highly_variable_genes(adata: AnnData,
                                 means: List[float],
                                 variances: List[float],
                                 max_genes: Optional[int] = 3000,
                                 inplace: bool = True,
                                 ):
    # dispersions
    dispersions = np.log(variances / (means + 1e-10) + 1)

    # z-scores
    z_scores = compute_z_scores(dispersions)

    # Loess regression
    x = np.log(means+1)
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

    df['gene'] = adata.var_names[qval_indices]
    df['p_adj'] = p_adj[qval_indices]
    df['z_score'] = z_scores[qval_indices]
    df['means_log'] = x[qval_indices]
    df['dispersions_log'] = dispersions[qval_indices]
    
    df.sort_values(
        ['p_adj', 'z_score'],
        ascending=[True, False],
        na_position='last',
        inplace=True,
    )

    hvg_list = df['gene'][0:max_genes].tolist()

    hvg_dict = {}
    for iter_gene in adata.var_names:
        if iter_gene in hvg_list:
            hvg_dict[iter_gene] = True
        else:
            hvg_dict[iter_gene] = False

    if inplace:
        adata.uns['hvg'] = {'flavor': 'hicat'}
        adata.var['highly_variable'] = pd.Series(data=hvg_dict, index=hvg_dict.keys())
        adata.var['p_adj'] = df['p_adj'].values
        adata.var['z_score'] = df['z_score'].values
        adata.var['means_log'] = df['means_log'].values
        adata.var['dispersions_log'] = df['dispersions_log'].values
#       adata.save() # need to be able to save if in backed mode.
    else:
        df = df[0:max_genes]
        df['highly_variable'] = True
        return df

