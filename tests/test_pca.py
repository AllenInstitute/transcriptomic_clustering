import os
import pytest

import numpy as np
import pandas as pd
import scipy
import anndata as ad
import scanpy as sc
import transcriptomic_clustering as tc

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

@pytest.fixture
def tasic():
    normalized_data = sc.read_h5ad(os.path.join(DATA_DIR, "input", "input_normalize_result.h5"))
    selected_genes = pd.read_csv(os.path.join(DATA_DIR, "input", "input_select_genes.csv"))['x'].to_list()
    selected_cells = pd.read_csv(os.path.join(DATA_DIR, "input", "input_select_cells.csv"))['x'].to_list()
    pcs_r = pd.read_csv(os.path.join(DATA_DIR, "output", "pca_rot.csv")).iloc[:, 1:].to_numpy()
    return {
        'adata': normalized_data,
        'selected_genes': selected_genes,
        'selected_cells': selected_cells,
        'pcs': pcs_r
    }

def test_pca_tasic(tasic):
    set_selected_cells = set(tasic['selected_cells'])
    cell_mask = [i for i, obs in enumerate(tasic['adata'].obs['cells']) if obs in set_selected_cells]

    pcs_tc, _, _ = tc.pca(
        tasic['adata'],
        cell_select=cell_mask, gene_mask=tasic['selected_genes'],
        n_comps=5, svd_solver='arpack'
    )
    pcs_tc = pcs_tc.T
    
    cos_siml = []
    for i_pc in range(pcs_tc.shape[1]):
        pc_tc = pcs_tc[:, i_pc]
        pc_r = tasic['pcs'][:, i_pc]
        # cos_siml.append(np.dot(pc_tc, pc_r) / (np.linalg.norm(pc_tc) * np.linalg.norm(pc_r)))
        cos_siml.append(1 - scipy.spatial.distance.cosine(pc_tc, pc_r))
    
    # 1 or -1 equally acceptable
    cos_siml = np.abs(cos_siml)
    np.testing.assert_allclose(cos_siml, np.ones(cos_siml.shape))

def test_pca_auto(tasic):
    set_selected_cells = set(tasic['selected_cells'])
    cell_mask = [i for i, obs in enumerate(tasic['adata'].obs['cells']) if obs in set_selected_cells]

    pcs_tc, _, _ = tc.pca(
        tasic['adata'],
        cell_select=cell_mask, gene_mask=tasic['selected_genes'],
        n_comps=5, random_state=1,
    )
    pcs_tc = pcs_tc.T

    cos_siml = []
    for i_pc in range(pcs_tc.shape[1]):
        pc_tc = pcs_tc[:, i_pc]
        pc_r = tasic['pcs'][:, i_pc]
        # cos_siml.append(np.dot(pc_tc, pc_r) / (np.linalg.norm(pc_tc) * np.linalg.norm(pc_r)))
        cos_siml.append(1 - scipy.spatial.distance.cosine(pc_tc, pc_r))

    # 1 or -1 equally acceptable
    cos_siml = np.abs(cos_siml)
    np.testing.assert_allclose(cos_siml, np.ones(cos_siml.shape), rtol=1e-4)


def test_pca_chunked(tasic):
    set_selected_cells = set(tasic['selected_cells'])
    cell_mask = [i for i, obs in enumerate(tasic['adata'].obs['cells']) if obs in set_selected_cells]

    tasic['adata'] = sc.read_h5ad(os.path.join(DATA_DIR, "input", "input_normalize_result.h5"), 'r')
    pcs_tc, _, _ = tc.pca(
        tasic['adata'],
        cell_select=cell_mask, gene_mask=tasic['selected_genes'],
        n_comps=5, chunk_size=50,
    )
    pcs_tc = pcs_tc.T

    cos_siml = []
    for i_pc in range(pcs_tc.shape[1]):
        pc_tc = pcs_tc[:, i_pc]
        pc_r = tasic['pcs'][:, i_pc]
        # cos_siml.append(np.dot(pc_tc, pc_r) / (np.linalg.norm(pc_tc) * np.linalg.norm(pc_r)))
        cos_siml.append(1 - scipy.spatial.distance.cosine(pc_tc, pc_r))

    # 1 or -1 equally acceptable
    cos_siml = np.abs(cos_siml)
    np.testing.assert_allclose(cos_siml, np.ones(cos_siml.shape), rtol=2e-1)