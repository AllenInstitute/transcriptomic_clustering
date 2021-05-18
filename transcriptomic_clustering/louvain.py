from typing import Optional, Tuple

from scanpy import AnnData
from scanpy.external.tl import phenograph


def cluster_louvain(
    adata: AnnData,
    k: int,
    annotate: bool = False,
    **kwargs):
    """
    Uses the louvain phenograph algorithm to calculate clusters in the given adata.
    Out of concern for memory the annotate behavior and behavior on AnnData of the
    scanpy pheongraph function is overridden.

    Parameters
    -----------
    adata: an AnnData or array of data to cluster
    k: number of nearest neighbors to use during graph construction
    annotate: if True, community labels are added to adata as observations. Otherwise
             all scanpy pheongraph outputs are returned.

   Returns
   -----------
   cluster_by_obs: an array of community labels for each cell in adata, in order
   obs_by_cluster: a map of community labels to lists of cell indices in that community in order
   graph: the calculated adjacency graph on the adata
   q: the maximum modularity of the final clustering 
    """
    if 'copy' in kwargs:
        del kwargs['copy']

    cluster_by_obs, graph, q = phenograph(
        adata=adata.X,
        k=k,
        **kwargs
    )

    obs_by_cluster = {}
    for i in range(len(cluster_by_obs)):
        cluster = cluster_by_obs[i]
        if cluster not in obs_by_cluster:
            obs_by_cluster[cluster] = []
        obs_by_cluster[cluster].append(i)

    if annotate:
        adata.obs['pheno_louvain'] = cluster_by_obs

    return cluster_by_obs, obs_by_cluster, graph, q