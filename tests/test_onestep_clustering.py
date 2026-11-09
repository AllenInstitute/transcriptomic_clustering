import numpy as np
import pytest

from transcriptomic_clustering.onestep_clustering import OnestepKwargs, onestep_clust


@pytest.mark.parametrize("latent", [False, True])
def test_onestep_real_separated_clusters(separated_expression, latent):
    data = separated_expression
    if latent:
        data.obsm["latent"] = np.column_stack(
            [np.repeat([0., 100.], 30), np.tile(np.arange(30), 2) / 100]
        )
    kwargs = OnestepKwargs(
        highly_variable_kwargs={"max_genes": 40},
        pca_kwargs={"n_comps": 5},
        filter_pcs_kwargs={"method": "elbow", "max_pcs": 3},
        latent_kwargs={"latent_component": "latent"} if latent else {},
        cluster_louvain_kwargs={"k": 5},
        merge_clusters_kwargs={"n_markers": 5},
    )
    clusters, markers = onestep_clust(data, kwargs, random_seed=42)
    assert {frozenset(c) for c in clusters} == {
        frozenset(range(30)), frozenset(range(30, 60))
    }
    assert markers
    assert markers <= {f"gene_{i}" for i in range(20)}
    assert len(set(np.concatenate(clusters))) == data.n_obs
