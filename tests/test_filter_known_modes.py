import pytest

import numpy as np
import pandas as pd
import scipy as scp
import scanpy as sc
import anndata as ad

import transcriptomic_clustering as tc



def test_filter_known_modes():
    n_obs = 5000
    n_vars = 300
    n_pcs = 50

    obs_list = [f'obs_{i}' for i in range(n_obs)]
    pcs_list = [f'pcs_{i}' for i in range(n_pcs)]

    pcs = scp.stats.ortho_group.rvs(dim=n_vars)
    pcs = pcs[:, 0:n_pcs]

    X = np.random.rand(n_obs, n_vars) * 1000
    X -= X.mean(axis=0)
    X = X @ pcs

    projected_adata = ad.AnnData(X, obs=obs_list, var=pcs_list)

    knowns = 0.95 * X[:,[1,7,14]] + 0.05 * X[:,[5,10,1]]

    df_kns = pd.DataFrame(knowns, index=obs_list)

    filtered_proj_data, rm_mask = tc.filter_known_modes(projected_adata[::2,:], df_kns)
    expected_rm_mask = np.zeros((n_pcs,), dtype=bool)
    expected_rm_mask[[1,7,14]] = 1

    assert all(rm_mask == expected_rm_mask)




