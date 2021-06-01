from typing import Optional, Tuple, List, Dict, Union, Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import sys


def vec_chisq_test(cl1_ncells_per_gene: np.array(float),
                   cl1_ncells: int,
                   cl2_ncells_per_gene: np.array(float),
                   cl2_ncells: int,
                   chisq_threshold: Optional[float] = 15):
    """
        Vectorized Chi-squared tests for differential gene detection.

        Parameters
        ----------
        cl1_ncells_per_gene: a vector with the number of cells in the 1st cluster with detection of each gene
        cl1_ncells: an integer value with the total number of cells in the 1st cluster
        cl2_ncells_per_gene: a vector with the number of cells in the 2nd cluster with detection of each gene
        cl2_ncells: an integer value with the total number of cells in the 2nd cluster
        chisq_threshold: a threshold for keeping meaningful chi squared statistics

        Returns
        -------
        p_vals: a numpy array of p-values with detection of each gene

    """
    n_genes = cl1_ncells_per_gene.shape[0]

    eps = sys.float_info.epsilon

    cl1_present = cl1_ncells_per_gene + eps
    cl1_v1 = cl1_ncells - cl1_present + 2*eps
        
    cl2_present = cl2_ncells_per_gene + eps
    cl2_v2 = cl2_ncells - cl2_present + 2*eps

    observed = np.array([cl1_present, cl1_v1, cl2_present, cl2_v2])
    ncells = cl1_ncells + cl2_ncells

    present = cl1_present + cl2_present
    absent = ncells - present
    expected = np.array([present*cl1_ncells, absent*cl1_ncells, present*cl2_ncells, absent*cl2_ncells])/ncells + eps
    
    p_vals = np.ones(n_genes)
    for i in range(n_genes):
        chi_squared_stat = max(0,(((abs(observed[:,i]-expected[:,i])-0.5)**2)/expected[:,i]).sum())
        if chi_squared_stat < chisq_threshold:
            p_vals[i] = 1 - stats.chi2.cdf(x=chi_squared_stat, df=1)
    
    return p_vals

def de_pair_chisq(pair: tuple, 
                  cl_present: Union[pd.DataFrame, pd.Series],
                  cl_means: Union[pd.DataFrame, pd.Series],
                  cl_size: Dict[Any, int],
                  chisq_threshold: Optional[float] = 15) -> pd.DataFrame:
    """
        Perform pairwise differential detection tests using Chi-Squared test for a single pair of clusters.

        Parameters
        ----------
        pair: a tuple of length 2 specifying which clusters to compare
        cl_present: a data frame of gene detection proportions (genes x clusters) 
        cl_means: a data frame of normalized mean gene expression values (genes x clusters)
        cl.size: a dict of cluster sizes
        chisq_threshold: a threshold for keeping meaningful chi squared statistics

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

    p_vals = vec_chisq_test(cl_present_sorted[first_cluster].to_numpy()*cl_size[first_cluster], 
                            cl_size[first_cluster],
                            cl_present_sorted[second_cluster].to_numpy()*cl_size[second_cluster],
                            cl_size[second_cluster],
                            chisq_threshold=chisq_threshold)
    
    rejected,p_adj = fdrcorrection(p_vals)

    lfc = cl_means_sorted[first_cluster].to_numpy() - cl_means_sorted[second_cluster].to_numpy()

    de_statistics_chisq = pd.DataFrame(
        {
            "gene": cl_present.index.to_list(),
            "p_adj": p_adj,
            "p_value": p_vals,
            "lfc": lfc,
            "meanA": cl_means_sorted[first_cluster].to_numpy(),
            "meanB": cl_means_sorted[second_cluster].to_numpy(),
            "q1": cl_present_sorted[first_cluster].to_numpy(),
            "q2": cl_present_sorted[second_cluster].to_numpy()
        }
    )
    
    de_statistics_chisq.set_index("gene")

    return de_statistics_chisq

