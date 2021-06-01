from typing import Any, Tuple, Dict, List, Optional
import anndata as ad
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
import logging
from collections import defaultdict
import warnings
import transcriptomic_clustering as tc

def merge_clusters(
        adata_norm: ad.AnnData,
        adata_reduced: ad.AnnData,
        cluster_assignments: Dict[Any, np.ndarray],
        cluster_by_obs: np.ndarray,
        min_cluster_size: Optional[int] = 4,
        k: Optional[int] = 2,
        low_th: Optional[int] = 1,
        de_method: Optional[str] = 'chisq',
        score_th: Optional[int] = 150,
        max_sampled_cells: Optional[int] = 300,
        markers: Optional[int] = 50, # If none, don't return makers. Otherwise, number of markers
        chunk_size: Optional[int] = None
):
    # TODO: Add doc str
    # TODO: Add all the thresholds

    # Calculate cluster means on reduced space
    cl_means_reduced, _ = tc.get_cluster_means(adata_reduced, cluster_assignments, cluster_by_obs, chunk_size, low_th)

    # Merge small clusters
    merge_small_clusters(cl_means_reduced, cluster_assignments, min_cluster_size)

    # Create new cluster_by_obs based on updated cluster assignments
    cluster_by_obs = np.zeros((adata_norm.shape[0],))
    for cl_id, idxs in cluster_assignments.items():
        cluster_by_obs[idxs] = cl_id

    # Calculate cluster means on normalized data
    cl_means, present_cl_means = tc.get_cluster_means(adata_norm, cluster_assignments, cluster_by_obs, chunk_size, low_th)

    # Merge remaining clusters by differential expression
    merge_clusters_by_de(cluster_assignments, cl_means, present_cl_means, cl_means_reduced, k, de_method, score_th)

    # TODO: Compute marker differential expressed genes based on function param, top cluster markers, etc.
    # Calculate de for all pairs of clusters
    # Cells should be sampled based on `max_sampled_cells`- equivalent to max.cl.size in R


    # Returns
    # - updated cluster assignments
    # - differentially expressed genes
    # - final cluster pairwise de.score
    # - top cluster pairwise markers



def merge_two_clusters(
        cluster_assignments: Dict[Any, np.ndarray],
        label_source: Any,
        label_dest: Any,
        cluster_means: pd.DataFrame,
        present_cluster_means: pd.DataFrame=None
):
    """
    Merge source cluster into a destination cluster by:
    1. updating cluster means
    2. updating mean of expressions present if not None
    3. updating cluster assignments

    Parameters
    ----------
    cluster_assignments:
        map of cluster label to cell idx belonging to cluster
    label_source:
        label of cluster being merged
    label_dest:
        label of cluster merged into
    cluster_means:
        dataframe of cluster means indexed by cluster label
    present_cluster_means:
        dataframe of cluster means indexed by cluster label filtered by low_th

    Returns
    -------
    """

    merge_cluster_means(cluster_means, cluster_assignments, label_source, label_dest)

    if present_cluster_means is not None:
        merge_cluster_means(present_cluster_means, cluster_assignments, label_source, label_dest)

    # merge cluster assignments
    cluster_assignments[label_dest] += cluster_assignments[label_source]
    cluster_assignments.pop(label_source)


def merge_cluster_means(
        cluster_means: pd.DataFrame,
        cluster_assignments: Dict[Any, np.ndarray],
        label_source: Any,
        label_dest: Any
):
    """
    Merge source cluster into a destination cluster by:
    1. computing the updated cluster centroid (mean gene expression)
        of the destination cluster
    2. deleting source cluster after merged

    Parameters
    ----------
    cluster_means:
        dataframe of cluster means indexed by cluster label
    cluster_assignments:
        map of cluster label to cell idx belonging to cluster
    label_source:
        label of cluster being merged
    label_dest:
        label of cluster merged into

    Returns
    -------
    """

    # update cluster means:
    n_source = len(cluster_assignments[label_source])
    n_dest = len(cluster_assignments[label_dest])

    cluster_means.loc[label_dest] = (cluster_means.loc[label_source] * n_source +
                                     cluster_means.loc[label_dest] * n_dest
                                    ) / (n_source + n_dest)

    # remove merged cluster
    cluster_means.drop(label_source, inplace=True)


def pdist_normalized(
        X: np.ndarray
) -> np.ndarray:
    """
    Calculate similarity metric as (1 - pairwise_distance/max_distance)

    Parameters
    ----------
    X:
        An m by n array of m clusters in an n-dimensional space
    Returns
    -------
    similarity:
        measure of similarity
    """

    dist = squareform(pdist(X))
    dist_norm = dist / np.max(dist)

    return 1 - dist_norm


def calculate_similarity(
        cluster_means: pd.DataFrame,
        group_rows: List[Any],
        group_cols: List[Any]
) -> pd.DataFrame:
    """
    Calculate similarity measure between two cluster groups (group_rows, group_cols)
    based on cluster means (cluster centroids)
    If data has more than 2 dimensions use correlation coefficient as a measure of similarity
    else use normalized distance measure

    Parameters
    ----------
    cluster_means:
        Dataframe of cluster means with cluster labels as index
    group_rows:
        cluster group with clusters being merged
    group_cols:
        cluster group with destination clusters
    Returns
    -------
    similarity:
        array of similarity measure
    """

    cluster_labels_subset = set(group_rows + group_cols)
    means = cluster_means.loc[cluster_labels_subset]
    n_clusters, n_vars = means.shape

    if n_vars > 2:
        similarity_df = means.T.corr()
    else:
        similarity = pdist_normalized(means)
        similarity_df = pd.DataFrame(similarity,
                                     index=cluster_labels_subset,
                                     columns=cluster_labels_subset)

    np.fill_diagonal(similarity_df.values, np.nan)

    return similarity_df.loc[group_rows][group_cols]


def find_most_similar(
        similarity_df: pd.DataFrame,
) -> Tuple[Any, Any, float]:
    """

    Parameters
    ----------
    similarity_df:
        similarity metric between clusters
    Returns
    -------
    source_label, dest_label, max_similarity:
        labels of the source and destination clusters and their similarity value
    """


    similarity_df = similarity_df.transpose()

    similarity_sorted = similarity_df.unstack().sort_values(ascending=False).dropna()
    source_label, dest_label = similarity_sorted.index[0]
    max_similarity = similarity_sorted[(source_label, dest_label)]

    return source_label, dest_label, max_similarity


def find_small_clusters(
        cluster_assignments: Dict[Any, np.ndarray],
        min_size: int
) -> List[Any]:

    return [k for (k, v) in cluster_assignments.items() if len(v) < min_size]


def merge_small_clusters(
        cluster_means: pd.DataFrame,
        cluster_assignments: Dict[Any, np.ndarray],
        min_size: int,
):
    """
    Then merge small clusters (with size < min_size) iteratively as:

    1. calculate similarity between small and all clusters
    2. merge most-highly similar small cluster
    3. update list of small/all clusters
    4. go to 1 until all small clusters are merged

    Parameters
    ----------
    cluster_means:
        dataframe of cluster means indexed by cluster label
    cluster_assignments:
        map of cluster label to cell idx belonging to cluster
    min_size:
        smallest size that is not merged

    Returns
    -------
    """
    all_cluster_labels = list(cluster_assignments.keys())
    small_cluster_labels = find_small_clusters(cluster_assignments, min_size=min_size)

    while small_cluster_labels:
        similarity_small_to_all_df = calculate_similarity(
            cluster_means,
            group_rows=small_cluster_labels,
            group_cols=all_cluster_labels)

        source_label, dest_label, max_similarity = find_most_similar(
            similarity_small_to_all_df,
        )
        logging.info(f"Merging small cluster {source_label} into {dest_label} -- similarity: {max_similarity}")
        merge_two_clusters(cluster_assignments, source_label, dest_label, cluster_means)

        # update labels:
        small_cluster_labels = find_small_clusters(cluster_assignments, min_size=min_size)
        all_cluster_labels = list(cluster_assignments.keys())


def get_de_scores_for_pairs(
    pairs: Tuple[int, int],
    cluster_means: pd.DataFrame,
    present_cluster_means: pd.DataFrame,
    cl_size: Dict[Any, int],
    de_method: Optional[str] = 'chisq'
) -> Tuple[Tuple[int, int], float]:
    """
    Calculate the de score for pairs of clusters

    Parameters
    ----------
    pairs:
        pairs of clusters to get score for
    cluster_means:
        dataframe of cluster means indexed by cluster label
    present_cluster_means:
        dataframe of cluster means indexed by cluster label filtered by low_th
    cl_size:
        mapping of cluster label to cluster size
    de_method:
        method used for de calculation

    Returns
    -------
    scores:
        calculated de score for each pair of clusters
    """

    scores = []
    for pair in pairs:
        if de_method == 'chisq':
            # TODO: This function is still in PR
            de_stats = tc.de_pair_chisq(pair, present_cluster_means, cluster_means, cl_size)
        elif de_method == 'limma':
            raise NotImplementedError('limma is not implemented')
        else:
            raise NotImplementedError(f'{de_method} is not implemented')

        # Calculate de score
        # TODO: This function is not implemented yet
        score = tc.get_de_score(de_stats)

        # Create ((dst, src), score) tuples
        scores.append((pair, score))

    return scores


def merge_clusters_by_de(
    cluster_assignments: Dict[Any, np.ndarray],
    cluster_means: pd.DataFrame,
    present_cluster_means: pd.DataFrame,
    cluster_means_rd: pd.DataFrame,
    k: Optional[int] = 2,
    de_method: Optional[str] = 'chi-sqr',
    score_th: Optional[int] = 150
):
    """
    Merge clusters by the calculated gene differential expression score

    Parameters
    ----------
    cluster_assignments:
        map of cluster label to cell idx belonging to cluster
    cluster_means:
        dataframe of cluster means indexed by cluster label in a normalized space
    present_cluster_means:
        dataframe of cluster means indexed by cluster label filtered by low_th in a normalized space
    cluster_means_rd:
        dataframe of cluster means indexed by cluster label in a reduced space
    k:
        number of cluster neighbors
    de_method:
        method used for de calculation
    score_th:
        threshold of de score for merging

    Returns
    -------
    """

    cl_size = [{k: len(v)} for k, v in cluster_assignments.items()]

    while len(cluster_assignments.keys()) > 1:
        # Use updated cluster means in reduced space to get nearest neighbors for each cluster
        # Steps 1-3
        neighbor_pairs = get_k_nearest_clusters(cluster_means_rd, k)

        if len(neighbor_pairs) == 0:
            break

        # TODO: Step 4: Get DE for pairs based on de_method
        scores = get_de_scores_for_pairs(neighbor_pairs, cluster_means, present_cluster_means, cl_size, de_method)

        # Sort scores
        scores = sorted(scores, key=(lambda x: x[1]))

        # Peek at first score and if > threshold, they are all greater than threshold
        pair, score = scores[0]
        if score >= score_th:
            break

        # Merge pairs below threshold, skipping already merged clusters
        merged_clusters = set()
        for score_pair in scores:
            pair, score = score_pair

            if score >= score_th:
                break

            dst_label, src_label = pair

            if dst_label in merged_clusters or src_label in merged_clusters:
                continue

            logging.info(f"Merging cluster {src_label} into {dst_label} -- de score: {score}")

            # Update cluster means on reduced space
            merge_cluster_means(cluster_means_rd, cluster_assignments, src_label, dst_label)

            # Update cluster means and cluster assignments
            merge_two_clusters(cluster_assignments, src_label, dst_label, cluster_means, present_cluster_means)
            merged_clusters.append(src_label)
            merged_clusters.append(dst_label)

            # Merge cluster sizes
            cl_size[dst_label] += cl_size[src_label]
            cl_size.pop(src_label)


def get_k_nearest_clusters(
        cluster_means: pd.DataFrame,
        k: Optional[int] = 2
) -> List[Tuple[int, int]]:
    """
    Get k nearest neighbors for each cluster

    Parameters
    ----------
    cluster_means:
        dataframe of cluster means with cluster labels as index
    k:
        number of nearest neighbors

    Returns
    -------
    nearest_neighbors:
        list of cluster pairs
    """

    cluster_labels = list(cluster_means.index)

    if k >= len(cluster_labels):
        warnings.warn("k cannot be greater than or the same as the number of clusters. "
                          "Defaulting to 2.")
        k = 2

    similarity = calculate_similarity(
            cluster_means,
            group_rows=cluster_labels,
            group_cols=cluster_labels)

    similarity = similarity.unstack().dropna()

    # Get k nearest neighbors
    nearest_neighbors = set()
    for c in cluster_labels:
        # Sort similarities for a cluster
        sorted_similarities = similarity.loc[(c, )].sort_values(ascending=False)

        for i in range(k):
            neighbor_cl = sorted_similarities.index[i]

            # Make sure neighbor doesn't already exist
            if not (neighbor_cl, c) in nearest_neighbors:
                nearest_neighbors.add((c, neighbor_cl))

    return list(nearest_neighbors)


def get_cluster_assignments(
        adata: ad.AnnData,
        cluster_label_obs: str = "pheno_louvain"
) -> Dict[Any, np.ndarray]:
    """

    Parameters
    ----------
    adata:
        AnnData object with with obs including cluster label
    cluster_label_obs:
        cluster label annotations in adata.obs

    Returns
    -------
    cluster_assignments:
        map of cluster label to cell idx
    """
    if cluster_label_obs not in list(adata.obs):
        raise ValueError(f"column {cluster_label_obs} is missing from obs")

    cluster_assignments = defaultdict(list)
    for i, label in enumerate(adata.obs[cluster_label_obs]):
        cluster_assignments[label].append(i)

    return cluster_assignments
