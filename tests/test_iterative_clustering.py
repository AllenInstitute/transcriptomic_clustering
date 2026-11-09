import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as scp
import scanpy as sc
import anndata as ad
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose

import transcriptomic_clustering as tc
from transcriptomic_clustering.final_merging import FinalMergeKwargs
from transcriptomic_clustering.iterative_clustering import (
    manage_cluster_adata, create_filebacked_clusters, iter_clust,
    summarize_final_clusters,
)
from transcriptomic_clustering.onestep_clustering import OnestepKwargs, onestep_clust


@pytest.mark.parametrize("clustering", [onestep_clust, iter_clust])
@pytest.mark.parametrize("return_markers_df", [False, True])
@pytest.mark.parametrize("marker_kwargs, expected_markers", [
    ({}, set()),
    ({"n_markers": None}, None),
])
def test_single_cluster_leaf(clustering, marker_kwargs, expected_markers, return_markers_df):
    norm_adata = ad.AnnData(np.ones((8, 3), dtype=np.float32))
    norm_adata.obsm["latent"] = np.zeros((8, 2), dtype=np.float32)
    cell_indices = list(range(norm_adata.n_obs))
    onestep_kwargs = OnestepKwargs(
        latent_kwargs={"latent_component": "latent"},
        merge_clusters_kwargs=marker_kwargs,
    )
    kwargs = {"tmp_dir": Path(".")} if clustering is iter_clust else {}

    with patch.object(tc, "cluster_louvain", return_value=(
        np.zeros(norm_adata.n_obs), {0: cell_indices}, None, None
    )):
        clusters, markers = clustering(
            norm_adata, onestep_kwargs=onestep_kwargs, **kwargs
        )

    assert clusters == [cell_indices]
    assert markers == expected_markers

    merged_clusters, merged_markers = tc.final_merge(
        norm_adata, clusters, marker_genes=markers,
        return_markers_df=return_markers_df,
        final_merge_kwargs=FinalMergeKwargs(merge_clusters_kwargs=marker_kwargs),
    )
    assert merged_clusters == clusters
    if return_markers_df:
        assert isinstance(merged_markers, pd.DataFrame)
        assert merged_markers.empty
    else:
        assert merged_markers == expected_markers


def test_create_filebacked_clusters_chunked():
    with patch("transcriptomic_clustering.memory.estimate_chunk_size", return_value=5) as mock_bar:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            norm_path = tmp_dir /'norm.h5ad'
            ad.AnnData(np.random.rand(20,3)).write(norm_path)
            norm_adata = sc.read_h5ad(norm_path, backed='r')

            clusters = [
                np.asarray([0,1,2,9,10,11,12,19],dtype=int),
                np.asarray([3,4,5,13,14,15],dtype=int),
                np.asarray([6,7,8,16,17,18],dtype=int),
            ]
            adatas = []
            try:
                adatas = create_filebacked_clusters(norm_adata, clusters, tmp_dir=tmp_dir)
                for result, indices in zip(adatas, clusters):
                    assert_allclose(result.X[:], norm_adata[indices, :].X)
                    assert_frame_equal(result.obs, norm_adata.obs.iloc[indices])
            finally:
                for result in adatas:
                    result.file.close()
                norm_adata.file.close()


def test_create_filebacked_clusters_not_chunked():
    with patch("transcriptomic_clustering.memory.estimate_chunk_size", return_value=30) as mock_bar:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            norm_path = tmp_dir / 'norm.h5ad'
            ad.AnnData(np.random.rand(20,3)).write(norm_path)
            norm_adata = sc.read_h5ad(norm_path, backed='r')

            clusters = [
                np.asarray([0,1,2,9,10,11,12,19],dtype=int),
                np.asarray([3,4,5,13,14,15],dtype=int),
                np.asarray([6,7,8,16,17,18],dtype=int),
            ]
            adatas = []
            try:
                adatas = create_filebacked_clusters(norm_adata, clusters, tmp_dir=tmp_dir)
                for result, indices in zip(adatas, clusters):
                    assert_allclose(result.X[:], norm_adata[indices, :].X)
                    assert_frame_equal(result.obs, norm_adata.obs.iloc[indices])
            finally:
                for result in adatas:
                    result.file.close()
                norm_adata.file.close()


def test_manage_cluster_adata_in_memory():
    adata = ad.AnnData(np.random.rand(20,3))
    clusters = [
                np.asarray([0,1,2,9,10,11,12,19],dtype=int),
                np.asarray([3,4,5,13,14,15],dtype=int),
                np.asarray([6,7,8,16,17,18],dtype=int),
            ]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        adatas = manage_cluster_adata(adata, clusters, tmp_dir=tmp_dir)
    assert_allclose(adatas[0].X, adata[clusters[0],:].X)
    for adata in adatas:
        assert not adata.isbacked


def test_manage_cluster_adata_to_memory():
    with patch("transcriptomic_clustering.memory.estimate_chunk_size", return_value=30) as mock_bar:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            x = np.random.rand(20,3)
            norm_path = tmp_dir / 'norm.h5ad'
            ad.AnnData(x).write(norm_path)
            norm_adata = sc.read_h5ad(norm_path, backed='r')

            clusters = [
                        np.asarray([0,1,2,9,10,11,12,19],dtype=int),
                        np.asarray([3,4,5,13,14,15],dtype=int),
                        np.asarray([6,7,8,16,17,18],dtype=int),
                    ]
            adatas = manage_cluster_adata(norm_adata, clusters, tmp_dir=tmp_dir)
            assert_allclose(adatas[0].X, x[clusters[0],:])
            for adata in adatas:
                assert not adata.isbacked


def test_iter_clust():
    rng = np.random.default_rng(5)
    def fake_onestep(norm_adata, *args, **kwargs):
        # randomly create samples
        n_clusts = rng.integers(1,10)
        cluster_map = rng.integers(n_clusts, size=norm_adata.n_obs)
        clusters = [[] for i in range(n_clusts)]
        for i, cl_id in enumerate(cluster_map):
            clusters[cl_id].append(i)
        return (
            [np.asarray(cluster, dtype=int) for cluster in clusters],
            set(rng.choice(norm_adata.var.index.tolist()))
        )

    with patch("transcriptomic_clustering.iterative_clustering.onestep_clust", wraps=fake_onestep) as mock_bar:
        n_obs = 300
        norm_adata = ad.AnnData(np.random.rand(n_obs,3))
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            clusters, markers = iter_clust(norm_adata, tmp_dir=tmp_dir, min_samples=4)

        set_indxs = set()
        for samples in clusters:
            set_indxs.update(samples.tolist())
        assert set(range(n_obs)) == set(set_indxs)


@pytest.mark.parametrize("markers_enabled", [False, True])
def test_iter_clust_recursive_marker_aggregation(tmp_path, markers_enabled):
    data = ad.AnnData(np.ones((8, 3), dtype=np.float32))

    def onestep(adata, **kwargs):
        if adata.n_obs == 8:
            return [[0, 2, 4, 6], [1, 3, 5, 7]], {"root"} if markers_enabled else None
        return [[0, 1], [2, 3]], {"leaf"} if markers_enabled else None

    with patch("transcriptomic_clustering.iterative_clustering.onestep_clust",
               side_effect=onestep) as step:
        clusters, markers = iter_clust(
            data, tmp_path, min_samples=4,
            onestep_kwargs=OnestepKwargs(
                merge_clusters_kwargs={"n_markers": 2 if markers_enabled else None}
            ),
        )
    assert step.call_count == 3
    assert [list(c) for c in clusters] == [[0, 2], [4, 6], [1, 3], [5, 7]]
    assert markers == ({"root", "leaf"} if markers_enabled else None)


def test_summarize_final_clusters():
    X = np.asarray([
        [50, 50, 0, 0, 0, 0],
        [75, 75, 0, 0, 0, 0],
        [60, 60, 0, 0, 0, 0],
        [0, 0, 20, 20, 0, 0],
        [0, 0, 20, 25, 0, 0],
        [0, 0, 0, 0, 100, 110],
        [0, 0, 0, 0, 90, 100],
        [0, 0, 0, 0, 95, 105],
    ])
    genes = pd.DataFrame(index=[f'Gene_{i}' for i in range(X.shape[1])])
    cells = pd.DataFrame(index=[f'Cell_{i}' for i in range(X.shape[0])])
    norm_adata = ad.AnnData(X, var=genes, obs=cells)
    clusters = [np.asarray([0,1,2]), np.asarray([3,4]), np.asarray([5,6,7])]
    thresholds = {
        'q1_thresh': 0.5,
        'q2_thresh': 0.7,
        'cluster_size_thresh': 0,
        'qdiff_thresh': 0.7,
        'padj_thresh': 0.01,
        'lfc_thresh': 1.0
    }
    de_table, linkage, labels = summarize_final_clusters(
        norm_adata,
        clusters,
        thresholds,
        de_method='ebayes'
    )

    assert all(de_table['up_num'] == 2)
    assert all(labels == [1,2,3])
