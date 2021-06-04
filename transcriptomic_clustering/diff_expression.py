from typing import Optional, Tuple, List, Dict, Union, Any

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

import sys


def vec_chisq_test(pair: tuple,
                  cl_present: pd.DataFrame,
                  cl_size: Dict[Any, int]):
    """
        Vectorized Chi-squared tests for differential gene detection.

        Parameters
        ----------
        pair: a tuple of length 2 specifying which clusters to compare
        cl_present: a data frame of gene detection proportions (genes x clusters)
        cl_size: a dict of cluster sizes

        Returns
        -------
        p_vals: a numpy array of p-values with detection of each gene

    """
    first_cluster = pair[0]
    second_cluster = pair[1]

    cl1_ncells_per_gene = cl_present[first_cluster].to_numpy()*cl_size[first_cluster]
    cl1_ncells = cl_size[first_cluster]
    cl2_ncells_per_gene = cl_present[second_cluster].to_numpy()*cl_size[second_cluster]
    cl2_ncells = cl_size[second_cluster]

    n_genes = cl1_ncells_per_gene.shape[0]

    cl1_present = cl1_ncells_per_gene
    cl1_absent = cl1_ncells - cl1_present
        
    cl2_present = cl2_ncells_per_gene
    cl2_absent = cl2_ncells - cl2_present

    observed = np.array([cl1_present, cl1_absent, cl2_present, cl2_absent])

    p_vals = np.ones(n_genes)
    for i in range(n_genes):
        try:
            chi_squared_stat, p_value, dof, ex = stats.chi2_contingency(observed[:,i].reshape(2,2), correction=True)
            p_vals[i] = p_value
        except:
            print("chi2 exception catched, p value will be assigned to 1")
    
    return p_vals

def de_pair_chisq(pair: tuple, 
                  cl_present: Union[pd.DataFrame, pd.Series],
                  cl_means: Union[pd.DataFrame, pd.Series],
                  cl_size: Dict[Any, int]) -> pd.DataFrame:
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

    p_vals = vec_chisq_test(pair, 
                            cl_present_sorted,
                            cl_size)
    
    rejected,p_adj = fdrcorrection(p_vals)

    lfc = cl_means_sorted[first_cluster].to_numpy() - cl_means_sorted[second_cluster].to_numpy()

    q1 = cl_present_sorted[first_cluster].to_numpy()
    q2 = cl_present_sorted[second_cluster].to_numpy()
    qdiff = get_qdiff(q1, q2)

    de_statistics_chisq = pd.DataFrame(
        {
            "gene": cl_present.index.to_list(),
            "p_adj": p_adj,
            "p_value": p_vals,
            "lfc": lfc,
            "meanA": cl_means_sorted[first_cluster].to_numpy(),
            "meanB": cl_means_sorted[second_cluster].to_numpy(),
            "q1": q1,
            "q2": q2,
            "qdiff": qdiff,
        }
    )
    
    de_statistics_chisq.set_index("gene")

    return de_statistics_chisq


def filter_gene_stats(
    de_stats: pd.DataFrame,
    gene_type: str,
    cl1_size: float = None,
    cl2_size: float = None,
    q1_thresh: float = None,
    q2_thresh: float = None,
    min_cell_thresh: int = None,
    qdiff_thresh: float = None,
    padj_thresh: float = None,
    lfc_thresh: float = None

) -> pd.DataFrame:
    """
    Filter out differential expression summary stats
    for either up-regulated or down-regulated genes

    Parameters
    ----------
    de_stats:
        dataframe with stats for each gene
    gene_type:
    cl1_size:
        cluster size of the first cluster
    cl2_size:
        cluster size of the second cluster
    q1_thresh:
        threshold for proportion of cells expressing each gene in the first cluster
    q2_thresh:
        threshold for proportion of cells expressing each gene in the second cluster
    min_cell_thresh:
        threshold for min number of cells in cluster
    qdiff_thresh:
        threshold for qdiff
    padj_thresh:
        threshold for padj
    lfc_thresh:
        threshold for lfc

    Returns
    -------
    filtered dataframe
    """

    if gene_type == 'up-regulated':
        mask = de_stats['lfc'] > 0
        qa = 'q1'
        qb = 'q2'
        cl_size = cl1_size
    elif gene_type == 'down-regulated':
        mask = de_stats['lfc'] < 0
        qa = 'q2'
        qb = 'q1'
        cl_size = cl2_size
    else:
        raise ValueError(f"Invalid gene_type value {gene_type}. "
                         f"Allowed values include 'up-regulated' and 'down-regulated'. ")

    if padj_thresh:
        mask &= de_stats['p_adj'] < padj_thresh
    if lfc_thresh:
        mask &= abs(de_stats['lfc']) > lfc_thresh

    if q1_thresh:
        mask &= de_stats[qa] > q1_thresh
    if cl_size:
        mask &= de_stats[qa] * cl_size >= min_cell_thresh
    if q2_thresh:
        mask &= de_stats[qb] < q2_thresh
    if qdiff_thresh:
        mask &= abs(de_stats['qdiff']) > qdiff_thresh

    return de_stats.loc[mask]


def get_qdiff(q1, q2) -> np.array:
    """
    Calculate normalized difference between q1 and q2 proportions
    when q1=0 and q2=0, return qdiff=0
    Parameters
    ----------
    q1:
        proportion of cells expressing each gene  for the first cluster
    q2:
        proportion of cells expressing each gene  for the second cluster

    Returns
    -------
    qdiff statistic
    """
    qmax = np.maximum(q1, q2)
    qmax[qmax == 0] = np.nan
    q_diff = abs(q1 - q2) / qmax
    np.nan_to_num(q_diff, nan=0.0, copy=False)

    return q_diff


def calc_de_score(
    padj: np.ndarray
) -> float:
    """
    Calculate DE score for a group of cells from padj values

    Parameters
    ----------
    padj:
        adjusted p-value

    Returns
    -------
    differential expression score
    """
    de_score = np.sum(-np.log10(padj)) if len(padj) else 0

    return de_score
