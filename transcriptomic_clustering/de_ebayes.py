from typing import Optional, Tuple, List, Dict, Union, Any
from numpy.core.fromnumeric import var
from numpy.typing import ArrayLike

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from scipy.stats.mstats import winsorize
from statsmodels.stats.multitest import fdrcorrection



"""
Implements functions for calculating differential expression
through moderated t-statistics as defined in 
Smyth, 2004 and Phipson et al., 2016

This module greatly simplifies the full process of
- Create dummy coding design with each cluster as an experimental condition (no intercept)
- fit linear model for each gene, calculated coefficients, residuals, degrees of freedom, etc
- moderate gene expression residual variances using empirical bayes
- create a contrast and update fit for cluster pair of interest
- perform t-test on each contrast fit
by recognizing 
- coefficients are means of each cluster,
- variances and degrees of freedom can be calculated directly from mean and mean squared values

"""




def moderate_variance(
        variance: pd.DataFrame,
        df: int,
        winsor_limits: Optional[Tuple(float,float)]=(0.05,0.1),
    ):
    """
    Moderated variances 
    
    - Assume each gene's variance is sampled from a 
      scaled inverse chi-square prior distribution
      with degrees of freedom d0 and location s_0^2 (sigma_0 squared)
    - Winsorize variances if desired
    - Fit fDist to get prior variance
    - Get posterior variance from sample variance and prior variance 
    
    
    Parameters
    ----------
    variances: sample variances (index = gene)
    df: degrees of freedom
    winsor_limits: upper and lower decimal percentile limits for winsorize, or None to skip winsorizing
    
    Returns
    -------
    moderated (posterior) variances, prior variance, prior degrees of freedom
    """

    if winsor_limits is not None:
        var_winz = winsorize(variance, limits=winsor_limits)
        var_winz = pd.DataFrame(var_winz, index=variance.index, columns=variance.columns)
    else:
        var_winz = variance
    
    df, df_prior, var_prior, _ = stats.f.fit(var_winz, fdfn=df, floc=np.zeros(var_winz.shape))
    var_post = (df_prior * var_prior + df * var_winz) / (df + df_prior)

    return var_post, var_prior, df_prior


def get_cl_var(cl_mean: pd.DataFrame, cl_mean_sq: pd.DataFrame):
    """
    Computes variances as E[x^2] - E[x]^2
    """
    if not(cl_mean.index.equals(cl_mean_sq.index)):
        raise ValueError(f'cl_mean and cl_mean_sq must have same index')
    # TODO: may suffer from precision issues, may need different method
    return cl_mean_sq - cl_mean ** 2 


def get_linear_fit_vals(cl_var: pd.DataFrame, cl_size: Dict[Any, int]):
    """
    Directly computes sigma squared, degrees of freedom, and stdev_unscaled
    for a linear fit of clusters from cluster variances and cluster size
    """
    cl_size_v = np.asarray([cl_size[clust] for clust in cl_var.index])
    df = cl_size_v.sum() - len(cl_size_v)
    sigma_sq = cl_size_v @ cl_var / df
    stdev_unscaled = pd.DataFrame(1 / np.sqrt(cl_size_v), index=cl_var.index)
    return sigma_sq, df, stdev_unscaled


def de_pairs_ebayes_statistic(
        pairs: List[Tuple[Any, Any]],
        cl_means: pd.DataFrame,
        cl_means_sq: pd.DataFrame,
        cl_size: Dict[Any, int],
        winsor_limits: Optional[Tuple(float,float)]=(0.05,0.1),
    ):
    """
    Computes moderated t-statistics for pairs of cluster

    Steps:
        Get sigma squared and degrees of freedom as if a linear model was fit for each gene
        Moderate all gene variances (see moderate variances for details)
        Compute cluster pair t-test p-val for all genes using moderated variances
        Adjust cluster pair pvals
        Filter and compute descore for pair
    
    Parameters
    ----------
    pairs: list of pairs of cluster names
    cl_means: dataframe with index = cluster name, columns = genes,
              values = cluster mean gene expression (E[X])
    cl_means: dataframe with index = cluster name, columns = genes,
              values = cluster mean gene expression square (E[X^2])
    cl_size: dict of cluster name: number of observations in cluster
    winsor_limits: upper and lower decimal percentile limits
                   for winsorize gene variances, or None to skip winsorizing

    Returns
    -------
    Dataframe with index = cluster_pair and columns = de score
    """
    cl_vars = get_cl_var(cl_means, cl_means_sq)
    sigma_sq, df, stdev_unscaled = get_linear_fit_vals(cl_vars, cl_size)
    sigma_sq_post, var_prior, df_prior = moderate_variance(sigma_sq, df, winsor_limits=winsor_limits)

    for (cluster_a, cluster_b) in pairs:
        means_diff = cl_means.loc[cluster_a] - cl_means.loc[cluster_b]
        stdev_unscaled_comb = np.sqrt(np.sum(stdev_unscaled.loc[[cluster_a, cluster_b]] ** 2))
        
        df_total = df + df_prior
        df_pooled = np.sum(df)
        df_total = np.min(df_total, df_pooled)
        
        t_vals = -np.abs(means_diff / stdev_unscaled_comb / np.sqrt(sigma_sq_post))
        
        p_vals = np.ones(t_vals.shape)
        for i, t in enumerate(t_vals):
            p_vals[i] = 1 - stats.t.cdf(t, df_total)
        _, p_adj = fdrcorrection(p_vals)
        lfc = means_diff

        de_pair_stats = pd.DataFrame(index=cl_means.columns)
        de_pair_stats['p_value'] = p_vals
        de_pair_stats['p_adj'] = p_adj
        de_pair_stats['lfc'] = lfc
        
        # TODO: filter and compute de score, append to a pair list