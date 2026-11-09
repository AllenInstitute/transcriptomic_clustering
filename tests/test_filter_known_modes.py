import pytest

import numpy as np
import pandas as pd
import scipy as scp
import scanpy as sc
import anndata as ad
from pandas.testing import assert_frame_equal

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

    projected_adata = ad.AnnData(
        X,
        obs=pd.DataFrame(index=obs_list),
        var=pd.DataFrame(index=pcs_list)
    )

    knowns = 0.95 * X[:,[1,7,14]] + 0.05 * X[:,[5,10,1]]

    df_kns = pd.DataFrame(knowns, index=obs_list)

    filtered_proj_data = tc.filter_known_modes(projected_adata[::2,:], df_kns)
    expected_keep_mask = np.ones((n_pcs,), dtype=bool)
    expected_keep_mask[[1,7,14]] = 0

    assert_frame_equal(filtered_proj_data.var, projected_adata[::2, expected_keep_mask].var)


@pytest.fixture
def orthogonal_projection():
    values = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    return ad.AnnData(
        values, dtype=values.dtype,
        obs=pd.DataFrame({"mode": -values[:, 0]}, index=list("abcd")),
        var=pd.DataFrame(index=["remove", "keep"]),
    )


@pytest.mark.parametrize("kind", ["column", "series", "frame"])
def test_known_modes_align_rows(orthogonal_projection, kind):
    projected = orthogonal_projection
    known = projected.obs["mode"]
    if kind == "column":
        known = "mode"
    elif kind == "series":
        known = known.iloc[::-1]
    else:
        known = known.to_frame().reindex(list("dcbax"), fill_value=5)
        known["constant"] = 1
    result = tc.filter_known_modes(projected, known)
    assert result.var_names.tolist() == ["keep"]
    np.testing.assert_array_equal(result.X, projected.X[:, [1]])
    assert projected.n_vars == 2


@pytest.mark.parametrize("kind", ["none", "empty", "constant", "threshold"])
def test_known_modes_noop(orthogonal_projection, kind):
    projected = orthogonal_projection
    known = {
        "none": None,
        "empty": pd.DataFrame(index=projected.obs_names),
        "constant": pd.Series(1, index=projected.obs_names),
        "threshold": "mode",
    }[kind]
    result = tc.filter_known_modes(projected, known, similarity_threshold=1)
    np.testing.assert_array_equal(result.X, projected.X)


@pytest.mark.parametrize("kind,error,message", [
    ("absent", ValueError, "not found"),
    ("array", TypeError, "column name"),
    ("missing", ValueError, "missing"),
    ("duplicate", ValueError, "unique"),
    ("nan", ValueError, "finite"),
    ("infinity", ValueError, "finite"),
    ("text", ValueError, "numeric"),
])
def test_known_modes_invalid(orthogonal_projection, kind, error, message):
    projected = orthogonal_projection
    known = {
        "absent": "absent",
        "array": np.arange(4),
        "missing": pd.Series([1., 2.], index=list("ab")),
        "duplicate": pd.Series([1., 2., 3., 4.], index=list("aabc")),
        "nan": pd.Series([1., 2., np.nan, 4.], index=list("abcd")),
        "infinity": pd.Series([1., 2., np.inf, 4.], index=list("abcd")),
        "text": pd.Series(["x"] * 4, index=list("abcd")),
    }[kind]
    with pytest.raises(error, match=message):
        tc.filter_known_modes(projected, known)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_known_modes_invalid_threshold(orthogonal_projection, threshold):
    with pytest.raises(ValueError, match="between"):
        tc.filter_known_modes(orthogonal_projection, "mode", threshold)



