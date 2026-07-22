"""
Regression tests for:
1. pairwise_DEGs.py – import and basic execution (fixes unterminated docstring)
2. dimension_reduction.py pca() – correct selected-gene counting with a
   sparse/non-contiguous gene mask
"""
import numpy as np
import pandas as pd
import anndata as ad
import pytest
from scipy.sparse import csr_matrix

import transcriptomic_clustering as tc


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def small_adata():
    """
    A tiny AnnData with 10 cells x 8 genes and two clear clusters.
    Cluster 0: cells 0-4 highly express genes 0-3.
    Cluster 1: cells 5-9 highly express genes 4-7.
    """
    rng = np.random.default_rng(42)
    X = np.zeros((10, 8), dtype=np.float32)
    X[0:5, 0:4] = rng.uniform(8, 12, (5, 4))
    X[5:10, 4:8] = rng.uniform(8, 12, (5, 4))
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(10)])
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(8)])
    return ad.AnnData(csr_matrix(X), obs=obs, var=var)


# ---------------------------------------------------------------------------
# 1. pairwise_DEGs import + execution
# ---------------------------------------------------------------------------

def test_pairwise_degs_import():
    """Importing pairwise_DEGs must not raise a SyntaxError."""
    import transcriptomic_clustering.pairwise_DEGs as pdeg  # noqa: F401


def test_pairwise_degs_runs(small_adata):
    """pairwise_degs() must complete and return a set of gene names."""
    from transcriptomic_clustering.pairwise_DEGs import pairwise_degs

    # Three clusters so ebayes degrees-of-freedom > 0
    obs_by_cluster = {
        0: list(range(0, 4)),
        1: list(range(4, 7)),
        2: list(range(7, 10)),
    }
    thresholds = {
        'q1_thresh': 0.5,
        'q2_thresh': None,
        'cluster_size_thresh': 2,
        'qdiff_thresh': 0.7,
        'padj_thresh': 0.05,
        'lfc_thresh': 1.0,
        'score_thresh': 1,
        'low_thresh': 1,
        'min_genes': 1,
    }

    markers = pairwise_degs(
        adata_norm=small_adata,
        obs_by_cluster=obs_by_cluster,
        thresholds=thresholds,
        n_markers=4,
        de_method='ebayes',
        n_jobs=1,           # avoid spawning a large process pool in tests
    )

    assert isinstance(markers, set), "pairwise_degs must return a set"


# ---------------------------------------------------------------------------
# 2. PCA – correct gene count with a sparse/non-contiguous mask
# ---------------------------------------------------------------------------

def test_pca_gene_count_with_sparse_mask():
    """
    Build a mask that selects a small number of genes whose *positional indices*
    sum to a value much larger than the actual count.  The old bug used
    sum(vidx) instead of np.count_nonzero(vidx_bool), which inflated n_genes
    and could hide the component-count clamping.

    With 50 genes, selecting only genes at positions [40, 41, 42] gives:
      - correct count : 3
      - buggy sum     : 40+41+42 = 123   (would set n_genes=123)

    With 20 cells, max_comps = min(20, 3) - 1 = 2.
    Requesting n_comps=10 should be silently clamped to 2.
    """
    n_obs, n_vars = 20, 50
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame(index=[f"c{i}" for i in range(n_obs)])
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])
    adata = ad.AnnData(X, obs=obs, var=var)

    # Select only 3 genes at high positional indices so sum(vidx) >> count
    selected_genes = [f"g{i}" for i in [40, 41, 42]]
    n_selected = 3   # the correct count

    # Pass all cells explicitly so dimension_reduction can count them
    all_cells = list(range(n_obs))

    # Requesting more components than possible; should be clamped to min-1
    result = tc.pca(
        adata,
        cell_select=all_cells,
        gene_mask=selected_genes,
        n_comps=10,
        random_state=0,
    )
    components = result[0]   # DataFrame of shape (n_genes, n_comps)
    # n_comps must be clamped: max_comps = min(n_obs, n_selected) - 1 = 2
    expected_n_comps = min(n_obs, n_selected) - 1
    assert components.shape[1] == expected_n_comps, (
        f"Expected {expected_n_comps} components (clamped), "
        f"got {components.shape[1]}.  "
        f"This may indicate sum(vidx) was used instead of "
        f"np.count_nonzero(vidx_bool)."
    )
