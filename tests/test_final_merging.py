from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import transcriptomic_clustering as tc
from transcriptomic_clustering.final_merging import sample_clusters
from transcriptomic_clustering.final_merging import FinalMergeKwargs


@pytest.fixture
def adata():
    return ad.AnnData(
        np.arange(32, dtype=np.float32).reshape(8, 4),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(8)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(4)]),
    )


@pytest.mark.parametrize(
    "cluster_dict,n_samples,expected_cells",
    [
        ({"large": [0, 1, 2, 3], "equal": [4, 5], "small": [6]},
         2, ["cell_0", "cell_2", "cell_4", "cell_5", "cell_6"]),
        ({"small": [1, 3]}, 3, ["cell_1", "cell_3"]),
        ({"equal": [1, 3]}, 2, ["cell_1", "cell_3"]),
        ({"all": list(range(8))}, 8, [f"cell_{i}" for i in range(8)]),
        ({"large": [0, 1, 2, 3]}, 0, []),
        ({"empty": []}, 2, []),
        ({}, 2, []),
    ],
)
def test_sample_clusters_selects_only_sampled_cells(
    adata, cluster_dict, n_samples, expected_cells
):
    cell_mask = sample_clusters(
        adata, cluster_dict, n_samples_per_clust=n_samples, random_seed=123
    )

    expected_mask = pd.Series(
        adata.obs_names.isin(expected_cells), index=adata.obs_names
    )
    assert_series_equal(cell_mask, expected_mask)
    assert cell_mask.index[cell_mask].tolist() == expected_cells
    assert cell_mask.sum() == len(expected_cells)


@pytest.mark.parametrize(
    "marker_genes",
    [
        ["gene_3", "gene_1"],
        {"gene_3", "gene_1"},
        ["gene_2"],
        [],
        ["gene_0", "gene_1", "gene_2", "gene_3"],
    ],
)
def test_final_merge_passes_only_sampled_cells_and_marker_genes_to_pca(
    adata, marker_genes
):
    class StopAfterPCA(Exception):
        pass

    with patch.object(tc, "pca", side_effect=StopAfterPCA) as pca:
        with pytest.raises(StopAfterPCA):
            tc.final_merge(
                adata,
                [[0, 1, 2, 3], [4, 5], [6, 7]],
                marker_genes=marker_genes,
                n_samples_per_clust=2,
                random_seed=123,
            )

    pca.assert_called_once()
    assert pca.call_args.args[0] is adata
    cell_mask = pca.call_args.kwargs["cell_select"]
    gene_mask = pca.call_args.kwargs["gene_mask"]
    expected_cells = ["cell_0", "cell_2", "cell_4", "cell_5", "cell_6", "cell_7"]
    assert_series_equal(
        cell_mask,
        pd.Series(adata.obs_names.isin(expected_cells), index=adata.obs_names),
    )
    assert_series_equal(
        gene_mask,
        pd.Series(adata.var_names.isin(marker_genes), index=adata.var_names),
    )
    assert cell_mask.index[cell_mask].tolist() == expected_cells
    assert set(gene_mask.index[gene_mask]) == set(marker_genes)
    assert cell_mask.sum() == len(expected_cells)
    assert gene_mask.sum() == len(marker_genes)


@pytest.mark.parametrize("latent", [False, True])
def test_final_merge_real_marker_set(separated_expression, latent):
    data = separated_expression
    data.obsm["latent"] = np.column_stack(
        [np.repeat([0., 100.], 30), np.tile(np.arange(30), 2) / 100]
    )
    clusters, markers = tc.final_merge(
        data, [list(range(15)), list(range(15, 30)), list(range(30, 60))],
        marker_genes={f"gene_{i}" for i in range(20)},
        random_seed=42,
        final_merge_kwargs=FinalMergeKwargs(
            pca_kwargs={"n_comps": 3},
            filter_pcs_kwargs={"method": "elbow", "max_pcs": 2},
            latent_kwargs={"latent_component": "latent"} if latent else {},
            merge_clusters_kwargs={"n_markers": 5},
        ),
    )
    assert {frozenset(c) for c in clusters} == {
        frozenset(range(30)), frozenset(range(30, 60))
    }
    assert markers
    assert markers <= {f"gene_{i}" for i in range(20)}
