from typing import Optional, Tuple

from scanpy import AnnData
from scanpy.external.tl import phenograph
from pandas import Categorical


def cluster_louvain(
    adata: AnnData,
    k: int,
    inplace: bool = False,
    **kwargs):
    """
    Uses the louvain phenograph algorithm to calculate clusters in the given adata.
    Out of concern for memory the inplace behavior and behavior on AnnData of the
    scanpy pheongraph function is overridden.

    Parameters
    -----------
    adata: an AnnData or array of data to cluster
    k: number of nearest neighbors to use during graph construction
    inplace: if True, community labels are added to adata as observations. Otherwise
             all scanpy pheongraph outputs are returned.

   Returns
   -----------
   communities: an array of community labels for each cell in adata, in order
   graph: the calculated adjacency graph on the adata
   q: the maximum modularity of the final clustering 
    """
    pca_data = None

    if isinstance(adata, AnnData):
        pca_data = adata.X
    else:
        pca_data = adata

    if 'copy' in kwargs:
        del kwargs['copy']

    communities, graph, q = phenograph(
        adata=pca_data,
        k=k,
        **kwargs
    )

    if not inplace or not isinstance(adata, AnnData):
        return communities, graph, q
    else:
        adata.obs['pheno_louvain'] = Categorical(communities)