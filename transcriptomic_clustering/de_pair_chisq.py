from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

def de_pair_chisq(pair: Tuple(Any, Any), 
                  cl_present: Dict[Any, np.array],
                  cl_means: Dict[Any, np.array],
                  cl_size: Dict[Any, int],
                  genes: List[str]):
    """
        Perform pairwise differential detection tests using Chi-Squared test for a single pair of clusters.

        Parameters
        ----------
        pair: a dict of length 2 specifying which clusters to compare
        cl_present: a dict of gene detection proportions (genes x clusters) 
        cl_means: a dict of normalized mean gene expression values (genes x clusters)
        cl.size: a dict of cluster sizes
        genes: the genes to use for pairwise comparisons

        Returns
        -------
        a data frame with DE statistics:
            padj: p-values adjusted
            pval: p-values
            lfc: log fold change of mean expression values between the pair of clusters
            meanA: mean expression value for the first cluster in the pair
            meanB: mean expression value for the second cluster in the pair
            q1: proportion of cells expressing each gene for the first cluster
            q2: proportion of cells expressing each gene for the second cluster

    """
    eps = 1e-9

    if len(pair) < 2:
        raise ValueError("The pair must be a dict of length 2 specifying which clusters to compare")

    first_cluster = pair[0]
    second_cluster = pair[1]

    n_genes = len(genes)

    if len(cl_present[first_cluster])!=n_genes or len(cl_present[second_cluster])!=n_genes:
        raise ValueError("The number of genes are inconsistent in cl_present")

    if len(cl_means[first_cluster])!=n_genes or len(cl_means[second_cluster])!=n_genes:
        raise ValueError("The number of genes are inconsistent in cl_means")

    cl1 = cl_present[first_cluster]*cl_size[first_cluster] + eps
    cl1_total = cl_size[first_cluster]
    cl1_v1 = cl1_total - cl1 + 2*eps
        
    cl2 = cl_present[second_cluster]*cl_size[second_cluster] + eps
    cl2_total = cl_size[second_cluster]
    cl2_v2 = cl2_total - cl2 + 2*eps

    observed = np.array([cl1, cl1_v1, cl2, cl2_v2])
    total = cl1_total + cl2_total

    present = cl1 + cl2
    absent = total - present
    expected = np.array([present*cl1_total, absent*cl1_total, present*cl2_total, absent*cl2_total])/total + eps
    
    p_vals = np.ones(n_genes)
    chisq_threshold = 15
    for i in range(n_genes):
        chi_squared_stat = max(0,(((abs(observed[:,i]-expected[:,i])-0.5)**2)/expected[:,i]).sum())
        if chi_squared_stat < chisq_threshold:
            p_vals[i] = 1 - stats.chi2.cdf(x=chi_squared_stat, df=1)
    
    rejected,p_adj = fdrcorrection(p_vals)

    lfc = np.log2(cl_means[first_cluster]/(cl_means[second_cluster]+eps)+eps)
    lfc[abs(lfc)>np.floor(np.log2(1/eps))] = 0

    de_statistics_chisq = pd.DataFrame(
        {
            "gene": genes,
            "p_adj": p_adj,
            "p_val": p_vals,
            "lfc": lfc,
            "meanA": cl_means[first_cluster],
            "meanB": cl_means[second_cluster],
            "q1": cl_present[first_cluster],
            "q2": cl_present[second_cluster]
        }
    )
    
    de_statistics_chisq.set_index("gene")

    return de_statistics_chisq

