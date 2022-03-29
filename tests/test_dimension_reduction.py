import os
import pytest

import numpy as np
import pandas as pd
import scipy
import anndata as ad
import scanpy as sc
import transcriptomic_clustering as tc
from transcriptomic_clustering.dimension_reduction import (
    filter_components, filter_ev_ratios_zscore, filter_explained_variances_elbow
)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


@pytest.fixture
def tasic():
    normalized_data = sc.read_h5ad(os.path.join(DATA_DIR, "input", "input_normalize_result.h5"))
    selected_genes = pd.read_csv(os.path.join(DATA_DIR, "input", "input_select_genes.csv"))['x'].to_list()
    selected_cells = pd.read_csv(os.path.join(DATA_DIR, "input", "input_select_cells.csv"))['x'].to_list()
    pcs_r = pd.read_csv(os.path.join(DATA_DIR, "output", "pca_rot.csv"))
    pcs_r = pcs_r.set_index("Unnamed: 0")
    return {
        'adata': normalized_data,
        'selected_genes': selected_genes,
        'selected_cells': selected_cells,
        'pcs': pcs_r
    }

def test_pca_tasic(tasic):
    set_selected_cells = set(tasic['selected_cells'])
    cell_mask = [i for i, obs in enumerate(tasic['adata'].obs['cells']) if obs in set_selected_cells]

    pcs_tc_T = tc.pca(
        tasic['adata'],
        cell_select=cell_mask, gene_mask=tasic['selected_genes'],
        n_comps=5, svd_solver='arpack'
    )[0].T

    cos_siml = pcs_tc_T @ tasic['pcs']
    cos_siml = np.abs(cos_siml)
    np.testing.assert_allclose(cos_siml, np.eye(cos_siml.shape[0]), rtol=1e-7, atol=1e-7)

def test_pca_auto(tasic):
    set_selected_cells = set(tasic['selected_cells'])
    cell_mask = [i for i, obs in enumerate(tasic['adata'].obs['cells']) if obs in set_selected_cells]

    pcs_tc_T= tc.pca(
        tasic['adata'],
        cell_select=cell_mask, gene_mask=tasic['selected_genes'],
        n_comps=5, random_state=1,
    )[0].T

    cos_siml = pcs_tc_T @ tasic['pcs']
    cos_siml = np.abs(cos_siml)
    np.testing.assert_allclose(cos_siml, np.eye(cos_siml.shape[0]), rtol=1e-4, atol=1e-4)


def test_pca_chunked(tasic):
    set_selected_cells = set(tasic['selected_cells'])
    cell_mask = [i for i, obs in enumerate(tasic['adata'].obs['cells']) if obs in set_selected_cells]

    tasic['adata'] = sc.read_h5ad(os.path.join(DATA_DIR, "input", "input_normalize_result.h5"), 'r')
    pcs_tc_T = tc.pca(
        tasic['adata'],
        cell_select=cell_mask, gene_mask=tasic['selected_genes'],
        n_comps=5, chunk_size=50,
    )[0].T

    cos_siml = pcs_tc_T @ tasic['pcs']
    cos_siml = np.abs(cos_siml)
    np.testing.assert_allclose(cos_siml, np.eye(cos_siml.shape[0]), rtol=0.15, atol=0.15)


def test_filter_known_components():
    n_pcs = 20
    pcs = scipy.stats.ortho_group.rvs(dim=(2*n_pcs))
    pcs = pcs[:, 0:n_pcs]
    pc_list = [f'PC_{i:02d}' for i in range(pcs.shape[1])]
    gene_list = [f'Gene_{i:02d}' for i in range(pcs.shape[0])]
    df_pcs = pd.DataFrame(pcs, columns=pc_list, index=gene_list)
    df_kns = df_pcs.iloc[::-1, 5:7]

    keep_pcs_mask = tc.dimension_reduction.filter_known_components(df_pcs, df_kns)
    expected_mask = np.ones((n_pcs,), dtype=bool)
    expected_mask[5:7] = 0

    assert all(keep_pcs_mask == expected_mask)


def test_filter_ev_ratios_zscore():
    n_pcs = 50
    explained_variance_ratios = [1 / (i + 1) for i in range(n_pcs)]
    explained_variance_ratios.sort()
    explained_variance_ratios = explained_variance_ratios[::-1] / np.sum(explained_variance_ratios)
    keep_pcs_mask = tc.dimension_reduction.filter_ev_ratios_zscore(explained_variance_ratios, threshold=2)

    expected_mask = np.ones((n_pcs,), dtype=bool)
    expected_mask[2:] = 0
    assert all(keep_pcs_mask == expected_mask)


def test_filter_explained_variances_elbow():
    explained_variances = np.asarray([256,128,64,32,16,8,4,2,1])
    explained_variances.sort()
    explained_variances = explained_variances[::-1]
    keep_pcs_mask = tc.dimension_reduction.filter_explained_variances_elbow(explained_variances)

    expected_mask = np.ones(explained_variances.shape, dtype=bool)
    expected_mask[4:]=0
    assert all(keep_pcs_mask == expected_mask)


def test_filter_components():
    n_pcs = 20
    pcs = scipy.stats.ortho_group.rvs(dim=(2*n_pcs))
    pcs = pcs[:, 0:n_pcs]
    pc_list = [f'PC_{i:02d}' for i in range(pcs.shape[1])]
    gene_list = [f'Gene_{i:02d}' for i in range(pcs.shape[0])]
    df_pcs = pd.DataFrame(pcs, columns=pc_list, index=gene_list)
    df_kns = df_pcs.iloc[::-1, 5:7]

    explained_variances = np.asarray([1.2 ** i for i in range(n_pcs)])
    explained_variances.sort()
    explained_variances = explained_variances[::-1]
    explained_variance_ratios = explained_variances / explained_variances.sum()

    pcs_obtained = tc.dimension_reduction.filter_components(
        df_pcs,
        explained_variances,
        explained_variance_ratios,
        known_components=df_kns,
        method='elbow',
        max_pcs=2
    )
    assert pcs_obtained.columns.to_list() == ['PC_00', 'PC_01']
