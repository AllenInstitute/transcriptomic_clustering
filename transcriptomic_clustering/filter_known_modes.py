from typing import Optional, Union

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import logging

logger = logging.getLogger(__name__)

def filter_known_modes(
        projected_adata: ad.AnnData,
        known_modes: Optional[Union[str, pd.Series, pd.DataFrame]] = None,
        similarity_threshold: Optional[float] = 0.7):
    """
        Filters out principal components which correlate strongly with the known modes

        Parameters
        ----------
        projected_adata: adata projected into principal components space
        known_modes: obs column name, or numeric Series/DataFrame indexed by
            cell names. Rows are aligned to projected_adata.obs_names; extra
            rows are allowed, but missing/duplicate rows or nonfinite values
            are rejected. Columns are known modes.
        similarity_threshold: threshold of correlation coefficients

        Returns
        -------
        projected_adata: after filtering out correlated principal components

    """
    # If not provided, skip filtering.
    if known_modes is None:
        return projected_adata

    if isinstance(known_modes, str):
        if known_modes not in projected_adata.obs.columns:
            raise ValueError(f'{known_modes} not found in projected_adata.obs')
        known_modes = projected_adata.obs[known_modes]
    elif not isinstance(known_modes, (pd.Series, pd.DataFrame)):
        raise TypeError("known_modes must be an obs column name, Series or DataFrame")
    
    if isinstance(known_modes, pd.Series):
        known_modes = known_modes.to_frame()

    if not known_modes.index.is_unique:
        raise ValueError("known_modes must have unique cell names")
    if not projected_adata.obs_names.isin(known_modes.index).all():
        raise ValueError("known_modes is missing projected cell names")
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1")
    try:
        mat_kns = known_modes.reindex(projected_adata.obs_names).to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("known_modes must contain numeric values") from error
    if not np.isfinite(mat_kns).all():
        raise ValueError("known_modes must contain finite values")
    # Constant modes cannot correlate with a component.
    mat_kns = mat_kns[:, np.ptp(mat_kns, axis=0) > 0]
    if mat_kns.shape[1] == 0 or projected_adata.n_vars == 0:
        return projected_adata

    n_pcs = projected_adata.n_vars
    n_remove = mat_kns.shape[1]

    with np.errstate(divide='ignore', invalid='ignore'):
        corr_pc_ev = np.corrcoef(projected_adata.X, mat_kns, rowvar=False)[:n_pcs, -n_remove:]
    corr_pc_ev = np.nan_to_num(corr_pc_ev)

    corr_pcs = np.amax(abs(corr_pc_ev), axis=1)
    logger.debug(f'corr pcs: {corr_pcs}')

    rm_pcs_mask = corr_pcs > similarity_threshold
    logger.info(f'Removing {rm_pcs_mask.sum()} PC(s)')
    projected_adata = projected_adata[:,~rm_pcs_mask]

    return projected_adata
