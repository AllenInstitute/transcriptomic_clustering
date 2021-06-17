import pytest

import numpy as np
import pandas as pd
import scanpy as sc
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose

import transcriptomic_clustering as tc
from transcriptomic_clustering.de_ebayes import moderate_variances, get_linear_fit_vals, de_pairs_ebayes


@pytest.fixture
def thresholds():
    thresholds = {
        'q1_thresh': 0.5,
        'q2_thresh': 0.7,
        'min_cell_thresh': 4,
        'qdiff_thresh': 0.7,
        'padj_thresh': 0.01,
        'lfc_thresh': 1.0
    }
    return thresholds


@pytest.fixture
def cl_stats():
    index = ['a','b','c']
    columns = [f'gene{i}' for i in range(20)]
    tmp_var = np.asarray([20,25,10,15,25,30,110,30,15,20,20,25,30,15,25,20,20,25,15,10])
    vars = np.asarray([
        tmp_var,
        tmp_var + 5,
        tmp_var - 5,
    ])
    vars = pd.DataFrame(vars, index=index, columns=columns)
    
    tmp_mean = np.asarray([100, 150, 100, 90, 110, 50, 75, 125, 130, 140, 70, 150, 60, 50, 110, 100, 90, 120, 80, 100])
    means = np.asarray([
        tmp_mean,
        tmp_mean - 20,
        tmp_mean - 100
    ])
    means = pd.DataFrame(means, index=index, columns=columns)

    means_sq = vars + np.square(means)
    means_sq = pd.DataFrame(means_sq, index=index, columns=columns)

    tmp_present = np.asarray([0.25, 0.1, 0.5, 0.8, 0.8, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.001, 0.4, 0.4, 0.3, 0.7, 0.8, 0.9, 0.99, 0.95])
    present = np.asarray([
        tmp_present,
        tmp_present + tmp_present * 0.1,
        1 - tmp_present,
    ])
    present = pd.DataFrame(present, index=index, columns=columns)

    df = 15
    cl_size = {'a': 5,
               'b': 10,
               'c': 3}
    return {
        'cl_means': means,
        'cl_means_sq': means_sq,
        'cl_present': present,
        'cl_vars': vars,
        'cl_size': cl_size,
        'df': df
    }


def test_moderate_variances_not_robust():
    """Verify moderate variance matches R's squeezeVars"""
    index = [f'gene{i}' for i in range(20)]
    vars = np.asarray([20,25,10,15,25,30,110,30,15,20,20,25,30,15,25,20,20,25,15,10])
    vars = pd.DataFrame(vars, index=index)
    df = 15

    df_prior_expected = 18.92199
    var_prior_expected = 21.86946
    var_post_expected = pd.DataFrame(
        [21.04280, 23.25376, 16.62089, 18.83185, 23.25376, 25.46471, 60.84000, 25.46471,
         18.83185, 21.04280, 21.04280, 23.25376, 25.46471, 18.83185, 23.25376, 21.04280,
         21.04280, 23.25376, 18.83185, 16.62089],
        index=vars.index
    )

    var_post, var_prior, df_prior = moderate_variances(vars, df)

    assert_allclose(df_prior, df_prior_expected)
    assert_allclose(var_prior, var_prior_expected)
    assert_frame_equal(var_post, var_post_expected)


def test_get_linear_fit_vals(cl_stats):
    """Verify calculated sigma's and degrees of freedom match
    scrattch.hicat/dev_zy's simple lmFit()"""
    vars = cl_stats['cl_vars']
    cl_size = cl_stats['cl_size']
    df_expected = cl_stats['df']
    
    sigma_sq, df, stdev_unscaled = get_linear_fit_vals(vars, cl_size)

    stdev_unscaled_expected = pd.DataFrame(1 / np.sqrt(list(cl_size.values())), index=vars.index)
    sigma_sq_expected = pd.DataFrame([
        26 + 1/3, 32 + 1/3, 14 + 1/3, 20 + 1/3, 32 + 1/3,
        38 + 1/3, 134 + 1/3, 38 + 1/3, 20 + 1/3, 26 + 1/3,
        26 + 1/3, 32 + 1/3, 38 + 1/3, 20 + 1/3, 32 + 1/3,
        26 + 1/3, 26 + 1/3, 32 + 1/3, 20 + 1/3, 14 + 1/3],
        index=vars.columns
    )

    assert_frame_equal(stdev_unscaled, stdev_unscaled_expected)
    assert df == df_expected
    assert_frame_equal(sigma_sq, sigma_sq_expected)


def test_de_pairs_ebayes(cl_stats, thresholds):
    cl_means = cl_stats['cl_means']
    cl_vars = cl_stats['cl_vars']
    cl_present = cl_stats['cl_present']
    cl_size = cl_stats['cl_size']

    de_pairs = de_pairs_ebayes(
        [('a','b'),('a','c'),('b','c')],
        cl_means, cl_vars, cl_present, cl_size, thresholds
    )

    up_genes_expected = ['gene3', 'gene4', 'gene5', 'gene16', 'gene17', 'gene18', 'gene19']

    np.testing.assert_almost_equal(de_pairs.loc[('a','c')]['score'], 84.26030647753015, decimal=10)
    assert set(de_pairs.loc[('a','c')]['up_genes']) == set(up_genes_expected)
