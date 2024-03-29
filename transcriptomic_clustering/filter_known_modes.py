from typing import Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import logging

logger = logging.getLogger(__name__)

def filter_known_modes(
        projected_adata: ad.AnnData,
        known_modes: Union[pd.DataFrame, pd.Series],
        similarity_threshold: Optional[float] = 0.7):
    """
        Filters out principal components which correlate strongly with the known modes

        Parameters
        ----------
        projected_adata: adata projected into principal components space
        known_modes: adata projected into a principal component to remove
                        index is obs (cell) names, columns are known modes
        similarity_threshold: threshold of correlation coefficients

        Returns
        -------
        projected_adata: after filtering out correlated principal components

    """
    if isinstance(known_modes, pd.Series):
        known_modes = known_modes.to_frame()

    kns_index_sorted, _ = known_modes.align(projected_adata.obs, 'right', axis=0)
    mat_kns = kns_index_sorted.to_numpy()

    n_pcs = projected_adata.n_vars
    n_remove = mat_kns.shape[1]

    corr_pc_ev = np.corrcoef(projected_adata.X, mat_kns, rowvar=False)[:n_pcs, -n_remove:]

    corr_pcs = np.amax(abs(corr_pc_ev), axis=1)
    logger.debug(f'corr pcs: {corr_pcs}')

    rm_pcs_mask = corr_pcs > similarity_threshold
    logger.info(f'Removing {rm_pcs_mask.sum()} PC(s)')
    projected_adata = projected_adata[:,~rm_pcs_mask]

    return projected_adata

