from typing import Optional, Tuple, List, Dict, Union, Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

def de_pair_chisq(pair: tuple, 
                  cl_present: Union[pd.DataFrame, pd.Series],
                  cl_means: Union[pd.DataFrame, pd.Series],
                  cl_size: Dict[Any, int],
                  eps: Optional[float] = 1e-9,
                  chisq_threshold: Optional[float] = 15) -> pd.DataFrame:
    """
        Perform pairwise differential detection tests using Chi-Squared test for a single pair of clusters.

        Parameters
        ----------
        pair: a tuple of length 2 specifying which clusters to compare
        cl_present: a data frame of gene detection proportions (genes x clusters) 
        cl_means: a data frame of normalized mean gene expression values (genes x clusters)
        cl.size: a dict of cluster sizes

        Returns
        -------
        a data frame with differential expressions statistics:
            padj: p-values adjusted
            pval: p-values
            lfc: log fold change of mean expression values between the pair of clusters
            meanA: mean expression value for the first cluster in the pair
            meanB: mean expression value for the second cluster in the pair
            q1: proportion of cells expressing each gene for the first cluster
            q2: proportion of cells expressing each gene for the second cluster

    """

    if len(pair) != 2:
        raise ValueError("The pair must contain two cluster labels")

    first_cluster = pair[0]
    second_cluster = pair[1]

    if isinstance(cl_present, pd.Series):
        cl_present = cl_present.to_frame()
    if isinstance(cl_means, pd.Series):
        cl_means = cl_means.to_frame()

    cl_present_sorted = cl_present.sort_index()
    cl_means_sorted = cl_means.sort_index()

    if not cl_present_sorted.index.equals(cl_means_sorted.index):
        raise ValueError("The indices (genes) of the cl_means and the cl_present do not match")

    n_genes,_ = cl_present.shape

    cl1 = cl_present[first_cluster].to_numpy()*cl_size[first_cluster] + eps
    cl1_total = cl_size[first_cluster]
    cl1_v1 = cl1_total - cl1 + 2*eps
        
    cl2 = cl_present[second_cluster].to_numpy()*cl_size[second_cluster] + eps
    cl2_total = cl_size[second_cluster]
    cl2_v2 = cl2_total - cl2 + 2*eps

    observed = np.array([cl1, cl1_v1, cl2, cl2_v2])
    total = cl1_total + cl2_total

    present = cl1 + cl2
    absent = total - present
    expected = np.array([present*cl1_total, absent*cl1_total, present*cl2_total, absent*cl2_total])/total + eps
    
    p_vals = np.ones(n_genes)
    for i in range(n_genes):
        chi_squared_stat = max(0,(((abs(observed[:,i]-expected[:,i])-0.5)**2)/expected[:,i]).sum())
        if chi_squared_stat < chisq_threshold:
            p_vals[i] = 1 - stats.chi2.cdf(x=chi_squared_stat, df=1)
    
    rejected,p_adj = fdrcorrection(p_vals)

    lfc = cl_means[first_cluster].to_numpy() - cl_means[second_cluster].to_numpy()

    de_statistics_chisq = pd.DataFrame(
        {
            "gene": cl_present.index.to_list(),
            "p_adj": p_adj,
            "p_value": p_vals,
            "lfc": lfc,
            "meanA": cl_means[first_cluster].to_numpy(),
            "meanB": cl_means[second_cluster].to_numpy(),
            "q1": cl_present[first_cluster].to_numpy(),
            "q2": cl_present[second_cluster].to_numpy()
        }
    )
    
    de_statistics_chisq.set_index("gene")

    return de_statistics_chisq

