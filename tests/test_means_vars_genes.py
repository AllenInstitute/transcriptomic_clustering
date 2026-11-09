import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from transcriptomic_clustering import get_means_vars_genes


@pytest.mark.parametrize("as_sparse", [False, True])
@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("min_cells", [2, 6])
def test_means_vars_genes_storage_equivalence(tmp_path, as_sparse, backed, min_cells):
    values = np.array([[0., 1., 2.], [0., 2., 2.], [1., 1., 2.],
                       [0., 3., 2.], [0., 0., 2.]])
    matrix = sparse.csr_matrix(values) if as_sparse else values
    data = ad.AnnData(matrix, dtype=values.dtype)
    if backed:
        path = tmp_path / "normalized.h5ad"
        data.write_h5ad(path)
        data = ad.read_h5ad(path, backed="r")
    try:
        means, variances, gene_mask = get_means_vars_genes(
            data, low_thresh=1, min_cells=min_cells,
            chunk_size=2 if backed else None,
        )
    finally:
        if data.isbacked:
            data.file.close()
    expected_mask = [False, True, True] if min_cells == 2 else [False] * 3
    np.testing.assert_array_equal(gene_mask, expected_mask)
    counts = np.expm1(values)
    np.testing.assert_allclose(means, counts.mean(axis=0)[expected_mask])
    np.testing.assert_allclose(variances, counts.var(axis=0, ddof=1)[expected_mask])
