
import pandas as pd
import numpy as np
from transcriptomic_clustering.markers import select_marker_genes


def test_select_marker_genes():
    n_genes = 9
    n_clusters = 5
    genes = [f"Gene_{i}" for i in range(n_genes)]
    clusters = [i for i in range(n_clusters)]

    clust_means = pd.DataFrame(
        index=clusters,
        columns=genes,
        data=np.asarray([
            [10, 10, 10, 0, 0, 0, 0, 0, 0],
            [10, 10, 10, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 10, 10, 10, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 10, 10, 10],
            [10, 10, 10, 10, 10, 10, 10, 10, 10]
        ])
    )
    clust_vars = pd.DataFrame(
        index=clusters,
        columns=genes,
        data=np.asarray([
            [3, 3, 3, 5, 5, 5, 1, 1, 1],
            [3, 3, 3, 5, 5, 5, 1, 1, 1],
            [3, 3, 3, 5, 5, 5, 1, 1, 1],
            [3, 3, 3, 5, 5, 5, 1, 1, 1],
            [3, 3, 3, 5, 5, 5, 1, 1, 1]
        ])
    )
    # Clusters 0 and 1 are same so no marker genes
    # Cluster 3 and 4 don't meet present threshold, so no markers
    # Clust 0 and 2 have marker genes 0,1 3,4
    present_clust_means = pd.DataFrame(
        index=clusters,
        columns=genes,
        data=np.asarray([
            [0.8, 0.8, 0.4, 0, 0, 0, 0, 0, 0],
            [0.8, 0.8, 0.4, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.8, 0.8, 0.4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.4, 0.4, 0.4],
            [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        ])
    )
    cluster_assignments = {
        0: [1,2,3,24,25,26],
        1: [4,5,6,7,8,9],
        2: [10,11,12,13,14],
        3: [15,16,17,18],
        4: [19,20,21,22,23],
    }
    
    markers = select_marker_genes(
        cluster_assignments=cluster_assignments,
        cluster_means=clust_means,
        cluster_variances=clust_vars,
        present_cluster_means=present_clust_means,
        n_markers=6,
        thresholds={
            'q1_thresh': 0.5,
            'q2_thresh': None,
            'cluster_size_thresh': 3,
            'qdiff_thresh': 0.7,
            'padj_thresh': 0.01,
            'lfc_thresh': 1.0,
            'score_thresh': 150,
            'low_thresh': 1,
            'min_genes': 5
        },
        de_method='ebayes'
    )
    expected = {"Gene_0", "Gene_1", "Gene_3", "Gene_4"}
    assert markers == expected