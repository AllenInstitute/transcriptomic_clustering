from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc

def filter_known_modes(principal_components: pd.DataFrame,
                    known_modes: pd.DataFrame,
                    similarity_threshold: Optional[float] = 0.7):
    """
        Filtering out principal components which correlate strongly with the provided know modes

        Parameters
        ----------
        principal_components: pincipal components from dimension reduction,
                        index is gene names, columns are principal components
        known_modes: eigen vectors of gene expressions to filter out
                        index is gene names, columns are known modes
        similarity_threshold: threshold of correlation coefficients

        Returns
        -------
        principal_components: after filtering out correlated principal components

    """
    
    pcs_index_sorted = principal_components.sort_index()
    kns_index_sorted = known_modes.sort_index()

    if not pcs_index_sorted.index.equals(kns_index_sorted.index):
        raise ValueError("The indices (genes) of the principal components and the known modes do not match")

    mat_pcs = pcs_index_sorted.to_numpy()
    mat_kns = kns_index_sorted.to_numpy()

    n_pcs = mat_pcs.shape[1]
    n_evs = mat_kns.shape[1]

    corr_pc_ev = np.corrcoef(mat_pcs, mat_kns, rowvar=False)[:n_pcs, -n_evs:]

    corr_pcs = np.amax(abs(corr_pc_ev), axis=1)

    rm_pcs_mask = corr_pcs > similarity_threshold

    rm_pcs = pcs_index_sorted.columns[rm_pcs_mask]

    if not rm_pcs.empty:
        pcs_index_sorted.drop(rm_pcs, axis='columns', inplace=True)

    return pcs_index_sorted

