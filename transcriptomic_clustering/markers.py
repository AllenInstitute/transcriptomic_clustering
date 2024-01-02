from typing import Dict, List, Set, Literal, Any, Optional
import logging
from itertools import combinations

import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

import transcriptomic_clustering as tc

logger = logging.getLogger(__name__)


def select_marker_genes(
        cluster_assignments: Dict[Any, List],
        cluster_means: pd.DataFrame,
        cluster_variances: pd.DataFrame,
        present_cluster_means: pd.DataFrame,
        thresholds: Dict[str, Any],
        n_markers: int = 20,
        de_method: Optional[Literal['ebayes', 'chisq']] = 'ebayes',
) -> Set:
    """
    Selects n up genes and n down genes from the differentially expressed genes
    between each pair of clusters, and saves the combined set for all cluster pairs.

    Parameters
    ----------
    cluster_assignments:
        map of cluster label to cell idx belonging to cluster
    n_markers:
        number of markers to select from both differentially up and down genes
    de_method:
        method used for de calculation
    thresholds:
        threshold use de calculation
    
    Returns
    -------
    """
    cl_size = {k: len(v) for k, v in cluster_assignments.items()}

    cl_names = list(cluster_assignments.keys())
    cl_names.sort()

    n_cls = len(cl_names)
    n_pairs = n_cls * (n_cls - 1) / 2

    logger.debug(f"Generating markers for {n_pairs} pairs of clusters")

    # Since cl_names is sorted ascending, combinations will also be sorted
    # in ascending order, e.g. (1,2), (1,3), (2,3)
    # TODO - may need to calculate de for subset of pairs at a time
    thresholds = thresholds.copy()
    thresholds.pop('min_genes', None)
    thresholds.pop('low_thresh', None)
    thresholds.pop('score_thresh', None)
    neighbor_pairs = list(combinations(cl_names, 2))
    if de_method == 'ebayes':
        de_df = tc.de_pairs_ebayes(
            neighbor_pairs,
            cluster_means,
            cluster_variances,
            present_cluster_means,
            cl_size,
            thresholds,
        )
    elif de_method == 'chisq':
        de_df = tc.de_pairs_chisq(
            neighbor_pairs,
            cluster_means,
            present_cluster_means,
            cl_size,
            thresholds,
        )
    else:
        raise ValueError(f'Unknown de_method {de_method}, must be one of [chisq, ebayes]')
    
    markers = set()
    for pair, row in de_df.iterrows():
        markers.update(row.up_genes[:n_markers])
        markers.update(row.down_genes[:n_markers])
    
    return markers