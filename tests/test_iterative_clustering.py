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
from transcriptomic_clustering.iterative_clustering import (
    manage_cluster_adata, create_filebacked_clusters, iter_clust,
    summarize_final_clusters,
)


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
            adatas = create_filebacked_clusters(norm_adata, clusters, tmp_dir=tmp_dir)
            assert_allclose(adatas[0].X, norm_adata[clusters[0],:].X)


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
            adatas = create_filebacked_clusters(norm_adata, clusters, tmp_dir=tmp_dir)
            assert_allclose(adatas[0].X, norm_adata[clusters[0],:].X)


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
