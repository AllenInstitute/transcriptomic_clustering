import os
import pytest

import numpy as np
import pandas as pd
import scipy
import anndata as ad
import scanpy as sc
import transcriptomic_clustering as tc

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def test_pca_tasic():
    normalized_data = sc.read_h5ad(os.path.join(DATA_DIR, "input", "input_normalize_result.h5"))
    selected_genes = pd.read_csv(os.path.join(DATA_DIR, "input", "input_select_genes.csv"))['x'].to_list()
    selected_cells = pd.read_csv(os.path.join(DATA_DIR, "input", "input_select_cells.csv"))['x'].to_list()
    pcs_r = pd.read_csv(os.path.join(DATA_DIR, "output", "pca_rot.csv")).iloc[:, 1:].to_numpy()


    set_selected_cells = set(selected_cells)
    cell_mask = [i for i, obs in enumerate(normalized_data.obs['cells']) if obs in set_selected_cells]

    pcs_tc, _, _ = tc.pca(normalized_data, cell_select=cell_mask, gene_mask=selected_genes, n_comps=5, svd_solver='arpack')
    pcs_tc = pcs_tc.T
    
    cos_siml = []
    for i_pc in range(pcs_tc.shape[1]):
        pc_tc = pcs_tc[:, i_pc]
        pc_r = pcs_r[:, i_pc]
        # cos_siml.append(np.dot(pc_tc, pc_r) / (np.linalg.norm(pc_tc) * np.linalg.norm(pc_r)))
        cos_siml.append(1 - scipy.spatial.distance.cosine(pc_tc, pc_r))
    
    # 1 or -1 equally acceptable
    cos_siml = np.abs(cos_siml)
    np.testing.assert_allclose(cos_siml, np.ones(cos_siml.shape))
