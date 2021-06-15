import os
import functools
from datetime import datetime
import community as community_louvain
import networkx as nx
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from annoy import AnnoyIndex
from scanpy import AnnData
from scanpy.external.tl import phenograph
from scipy.sparse import csr_matrix
from math import log
import tempfile


def cluster_louvain_phenograph(
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

    if -1 in obs_by_cluster:
        max_cluster = max(list(obs_by_cluster.keys()))
        for obs in obs_by_cluster[-1]:
            obs_by_cluster[max_cluster + 1] = [obs]
            cluster_by_obs[obs] = max_cluster + 1
            max_cluster += 1
        
        del obs_by_cluster[-1]

    if annotate:
        adata.obs['pheno_louvain'] = cluster_by_obs

    return cluster_by_obs, obs_by_cluster, graph, q

def cluster_louvain(
    adata: AnnData,
    k: int,
    annotate: bool = False,
    nn_measure: str = 'euclidean',
    knn_method: str = 'annoy',
    louvain_method: str = 'taynaud',
    weighting_method: str = 'jaccard',
    annoy_trees: int = None,
    n_jobs: int = 1,
    resolution: float = 1.,
    annoy_index_filename: str = None,
    graph_filename: str = None,
    random_seed: int = None
):
    """
    Immitates the cluster_louvain_phenograph interface for consistency,
    but simply wraps separate K Nearest Neighbors functions and graph-parametered
    Louvain functions.

    Parameters
    -----------
    adata: an AnnData or array of data to cluster
    k: number of nearest neighbors to use during graph construction
    annotate: if True, community labels are added to adata as observations. Otherwise
              all scanpy pheongraph outputs are returned.
    nn_measure: metric to use for nearest neighbor evaluation. Can be "angular",
                "euclidean", "manhattan", "hamming", or "dot"
    knn_method: KNN method to use, currently can only be "annoy"
    louvain_method: Louvain method to use, currently can only be "taynaud"
    weighting_method: weighting method to use to use for nearest neighbors graph
                      Can currently be "jaccard" or "uniform".
    annoy_trees: Number of trees to use in annoy random forest index. More trees gives higher
                 precision but is more computationally expensive.
    n_jobs: Number of processes to use in any parallelizable steps
    resolution: Louvain resolution parameter, changes size of communities
    annoy_index_filename: File to store annoy index in, defaults to temp file
    graph_filename: File to store KNN graph AnnData in, if unset does not save graph
    random_seed: int to use as random seed 

    Returns
    -----------
    cluster_by_obs: an array of community labels for each cell in adata, in order
    obs_by_cluster: a map of community labels to lists of cell indices in that community in order
    graph: the calculated adjacency graph on the adata
    q: the maximum modularity of the final clustering 
    """
    if knn_method == 'annoy':
        nn_adata = get_annoy_knn(
            adata,
            k,
            nn_measure,
            weighting_method,
            annoy_trees,
            n_jobs,
            annoy_index_filename,
            graph_filename,
            random_seed
        )
    else:
        raise ValueError(f"{knn_method} is not a valid knn method! Only available method is annoy")

    if louvain_method == 'taynaud':
        cluster_by_obs, obs_by_cluster, q = get_taynaud_louvain(
            nn_adata,
            resolution,
            random_seed
        )
    else:
        raise ValueError(f"{louvain_method} is not a valid louvain method! Only available method is taynaud")

    if annotate:
        adata.obs['pheno_louvain'] = cluster_by_obs

    return cluster_by_obs, obs_by_cluster, nn_adata.X, q

def _uniform_csr_from_nn_dict(nn_dict):
    """
    Converts a dictionary mapping indices to lists of their nearest neighbor indices
    to a csr_matrix with ordered indices and uniform weights with value 1.

    Parameters
    -----------
    nn_dict: dictionary mapping indices to lists of their nearest neighbor indices

    Returns
    -----------
    uniform_csr: csr_matrix object with value 1 at each index (i,j) where j is a
                 nearest neighbor of i
    """
    n = len(nn_dict)
    k = len(nn_dict[0])

    csr_indices = []
    for i in range(n):
        csr_indices.extend(nn_dict.pop(i))

    csr_weights = [1 for i in range(n * k)]
    csr_indptr = [i * k for i in range(n)]

    return csr_matrix((csr_weights, csr_indices, csr_indptr), shape=(n, n), dtype=float)

def _calc_jaccard(idx, max_union, nn_set_dict):
    """
    Given a dictionary mapping indices to sets of their nearest neighbor indices,
    an index, and a constant value provided for computational convenience computes the
    jaccard coefficients for the nearest neighbors of observation idx. Returns both the index
    and the coefficients for post-parallelism re-ordering.

    Parameters
    -----------
    idx: index of the observation to evaluate
    max_union: 2°k, provided for computational brevity
    nn_dict: dictionary mapping indices to sets of their nearest neighbor indices

    Returns
    -----------
    idx: index of the evaluated observation
    nn_jaccard: list of jaccard coefficients for the observation's nearest neighbors
    """
    shared_neighbors = [float(len(nn_set_dict[idx] & nn_set_dict[j])) for j in nn_set_dict[idx]]
    return idx, [x / (max_union - x) for x in shared_neighbors]

def _jaccard_csr_from_nn_dict(nn_dict, n_jobs = 1):
    """
    Converts a dictionary mapping indices to lists of their nearest neighbor indices
    to a csr_matrix with ordered indices and jaccard coefficients as weights.

    Parameters
    -----------
    nn_dict: dictionary mapping indices to lists of their nearest neighbor indices
    n_joba: number of jobs to use in computing jaccard coefficients. Defaults to 1.

    Returns
    -----------
    jaccard_csr: csr_matrix object with jaccard coefficient of observations i and j at
                 index (i,j) where j is a nearest neighbor of i
    """
    n = len(nn_dict)
    k = len(nn_dict[0])

    pool = Pool(processes=n_jobs)
    weight_map = {}
    chunk_size = min(max(1, int(n / n_jobs)), 10000)
    nn_set_dict = {k: set(v) for k,v in nn_dict.items()}
    picklable_jaccard = functools.partial(_calc_jaccard, max_union = 2*k, nn_set_dict = nn_set_dict)
    for chunk_result in tqdm(pool.imap_unordered(picklable_jaccard, range(n), chunksize=chunk_size), total=n):
        weight_map[chunk_result[0]] = chunk_result[1]

    csr_weights = []
    csr_indices = []
    for i in range(n):
        csr_weights.extend(weight_map[i])
        csr_indices.extend(nn_dict[i])

    csr_indptr = [i * k for i in range(n)] + [n * k]
    return csr_matrix((csr_weights, csr_indices, csr_indptr), shape=(n, n), dtype=float)

def _search_nn_chunk(idx, k, vec_len, nn_measure, annoy_index_filename):
    """
    Given an observation index, loads the annoy index at annoy_index_filename
    and requests k nearest neighbors from that index.

    Parameters
    -----------
    idx: index of the observation to find nearest neighbors for
    k: number of nearest neighbors to find
    vec_len: length of the observation vectors
    nn_measure: distance metric used to evaluate nearest neighbors on this index
    annoy_index_filename: location of the file-cached annoy index

    Returns
    -----------
    idx: index of the evaluated observation
    nn_list: list of the indices of the k nearest neighbors to the observation at idx
    """
    ai = AnnoyIndex(vec_len, nn_measure)
    ai.load(annoy_index_filename)
    return idx, ai.get_nns_by_item(idx, k)

def _annoy_build_csr_nn_graph(
    data_matrix: np.array,
    annoy_index_filename: str,
    k: int,
    n_jobs: int = 1,
    nn_measure: str = 'euclidean',
    weighting_method: str = 'jaccard'
):
    """
    Given a matrix of observation data and the location of a file-cached annoy index on
    that data, generates a csr_matrix of the k nearest neighbors of each observation weighted
    by the method specified in weighting.

    Parameters
    -----------
    data_matrix: a matrix of observation data
    annoy_index_filename: location of the file-cached annoy index
    k: number of nearest neighbors to find
    n_jobs: number of processes to use while computing nearest neighbors
    nn_measure: distance metric used to calculate nearest neighbors
    weighting_method: weighting method to use to use for nearest neighbors graph.
                      Can currently be "jaccard" or "uniform".

    Returns
    -----------
    nn_csr: csr_matrix of weighted neaerest neighbors
    """
    n, vec_len = data_matrix.shape
    pool = Pool(processes=n_jobs)
    graph_map = {}
    chunk_size = min(max(1, int(n / n_jobs)), 10000)
    picklable_search = functools.partial(_search_nn_chunk, k=k, vec_len=vec_len, nn_measure=nn_measure, annoy_index_filename=annoy_index_filename)
    for chunk_result in tqdm(pool.imap_unordered(picklable_search, range(n), chunksize=chunk_size), total=n):
        graph_map[chunk_result[0]] = chunk_result[1]

    if weighting_method == 'jaccard':
        return _jaccard_csr_from_nn_dict(graph_map, n_jobs)
    if weighting_method == 'uniform':
        return _uniform_csr_from_nn_dict(graph_map, n_jobs)
    else:
        raise ValueError(f"{weighting_method} is not a valid weighting option! Must use jaccard or uniform")

def get_annoy_knn(
    adata: AnnData,
    k: int,
    nn_measure: str = 'euclidean',
    weighting_method: str = 'jaccard',
    annoy_trees: int = None,
    n_jobs: int = 1,
    annoy_index_filename: str = None,
    graph_filename: str = None,
    random_seed: int = None
):
    """
    Given an AnnData object of observation data, uses the annoy package to compute the k neareset neighbors
    of each observation. Returns an AnnData object with a nearest neighbor graph of the observations weighted
    by the method specified in weighting_method.

    Parameters
    -----------
    adata: an AnnData or array of data to cluster
    k: number of nearest neighbors to use during graph construction
    nn_measure: metric to use for nearest neighbor evaluation. Can be "angular",
                "euclidean", "manhattan", "hamming", or "dot"
    weighting_method: weighting method to use to use for nearest neighbors graph
                      Can currently be "jaccard" or "uniform".
    annoy_trees: Number of trees to use in annoy random forest index. More trees gives higher
                 precision but is more computationally expensive.
    n_jobs: Number of processes to use in any parallelizable steps
    annoy_index_filename: File to store annoy index in, defaults to temp file
    graph_filename: File to store KNN graph AnnData in, if unset does not save graph
    random_seed: int to use as random seed 

    Returns
    -----------
    nn_adata: An AnnData object with a weighted nearest neighbor graph of the observations
    """
    data_matrix = adata.X
    ai = AnnoyIndex(data_matrix.shape[1], nn_measure)
    if not annoy_index_filename:
        annoy_index_filename = os.path.join(tempfile.gettempdir(), f"annoy_index_{datetime.now().strftime('%Y%m%d%H%M%S')}")
    ai.on_disk_build(annoy_index_filename)
    if random_seed:
        ai.set_seed(random_seed)

    for i in range(data_matrix.shape[0]):
        ai.add_item(i, data_matrix[i])

    if not annoy_trees:
        annoy_trees = max(1, int(log(data_matrix.shape[0], 2)))

    ai.build(annoy_trees, n_jobs=n_jobs)

    csr_graph = _annoy_build_csr_nn_graph(data_matrix, annoy_index_filename, k, n_jobs, nn_measure, weighting_method)

    graph_adata = AnnData(csr_graph, obs=adata.obs, var=adata.obs)
    if graph_filename:
        graph_adata.write(graph_filename)
    return graph_adata

def get_taynaud_louvain(
    nn_adata: AnnData,
    resolution: float = 1.,
    random_seed: int = None
):
    """
    Given an AnnData object containing a weighted nearest neighbor graph, computes the
    louvain partition of the graph.

    Parameters
    -----------
    nn_adata: An AnnData object with a weighted nearest neighbor graph of the observations
    resolution: Louvain resolution parameter, changes size of communities
    random_seed: int to use as random seed 

    Returns
    -----------
    cluster_by_obs: an array of community labels for each cell in adata, in order
    obs_by_cluster: a map of community labels to lists of cell indices in that community in order
    q: the maximum modularity of the final clustering 
    """
    nn_coo = nn_adata.X.tocoo()
    G = nx.Graph()
    G.add_weighted_edges_from(list(zip(nn_coo.row, nn_coo.col, nn_coo.data)))

    partition = community_louvain.best_partition(G, resolution=resolution, random_state=random_seed)
    cluster_by_obs = [-1 for i in range(nn_coo.shape[0])]
    obs_by_cluster = {}
    for cell, cluster in partition.items():
        cluster_by_obs[cell] = cluster
        if cluster not in obs_by_cluster:
            obs_by_cluster[cluster] = [cell]
        else:
            obs_by_cluster[cluster].append(cell)

    q = community_louvain.modularity(partition, G)

    return cluster_by_obs, obs_by_cluster, q