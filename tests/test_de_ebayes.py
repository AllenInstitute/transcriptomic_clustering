import pytest

import numpy as np
import pandas as pd
import scanpy as sc
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose

import transcriptomic_clustering as tc
from transcriptomic_clustering.de_ebayes import moderate_variance, get_linear_fit_vals, de_pairs_ebayes



# def test_moderate_variance_robust():
#     """Verify moderate variance matches R's squeezeVars"""
#     tmp_var = [20,25,10,15,25,30,110,30,15,20,20,25,30,15,25,20,20,25,15,10] #[20, 25, 10, 15, 25, 30, 110] 
#     var = pd.DataFrame(tmp_var, index=[f'gene{i}' for i in range(len(tmp_var))])
#     df = 15

#     df_prior_expected = 208.6846
#     var_prior_expected = 21.63223
#     var_post_expected = pd.DataFrame(
#         [21.52278, 21.85807, 20.85219, 21.18749, 21.85807, 22.19337, 103.06222,
#          22.19337, 21.18749, 21.52278, 21.52278, 21.85807, 22.19337, 21.18749,
#          21.85807, 21.52278, 21.52278, 21.85807, 21.18749, 20.85219],
#         index=var.index
#     )

#     var_post, var_prior, df_prior = moderate_variance(var, df, winsor_limits=(0.05,0.1))
    
#     print(f'df_prior: {df_prior}, expected: {df_prior_expected}')
#     print(f'var_prior: {var_prior}, expected: {var_prior_expected}')
#     print(f'var_post: {var_post}, expected: {var_post_expected}')

#     assert_allclose(df_prior, df_prior_expected)
#     assert_allclose(var_prior, var_prior_expected)
#     assert_frame_equal(var_post, var_post_expected)


def test_moderate_variances_not_robust():
    """Verify moderate variance matches R's squeezeVars"""
    tmp_var = [20,25,10,15,25,30,110,30,15,20,20,25,30,15,25,20,20,25,15,10] 
    var = pd.DataFrame(tmp_var, index=[f'gene{i}' for i in range(len(tmp_var))])
    df = 15

    df_prior_expected = 18.92199
    var_prior_expected = 21.86946
    var_post_expected = pd.DataFrame(
        [21.04280, 23.25376, 16.62089, 18.83185, 23.25376, 25.46471, 60.84000, 25.46471,
         18.83185, 21.04280, 21.04280, 23.25376, 25.46471, 18.83185, 23.25376, 21.04280,
         21.04280, 23.25376, 18.83185, 16.62089],
        index=var.index
    )

    var_post, var_prior, df_prior = moderate_variance(var, df)

    assert_allclose(df_prior, df_prior_expected)
    assert_allclose(var_prior, var_prior_expected)
    assert_frame_equal(var_post, var_post_expected)

# def test_get_linear_fit_vals():
#     """Verify calculated sigma's and degrees of freedom match
#     scrattch.hicat/dev_zy's simple lmFit()"""
#     assert False

