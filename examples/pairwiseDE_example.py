# this example script is used to get pair-wise DEGs for all clusters given the normalized count and the cluster assignment 
import sys
import scanpy as sc
import pickle
from collections import defaultdict

sys.path.insert(1, '/allen/programs/celltypes/workgroups/rnaseqanalysis/dyuan/tool/transcriptomic_clustering/')
from transcriptomic_clustering.pairwise_DEGs import pairwise_degs

thresholds = {
    'q1_thresh': 0.5,
    'q2_thresh': None,
    'cluster_size_thresh': 10, 
    'qdiff_thresh': 0.7, 
    'padj_thresh': 0.05, 
    'lfc_thresh': 1, 
    'score_thresh': 100, 
    'low_thresh': 0.6931472, 
    'min_genes': 5
}

# reading in the adata that adata.X is already normalizedd
adata = sc.read('/allen/programs/celltypes/workgroups/rnaseqanalysis/dyuan/HMBA_analysis/iscANVI_mapping/troubleshoot/test_data/adata_query_MHGlut.h5ad')

# reading in the clustering results, a list of lists of cell indices in adata
clusters_pth = '/allen/programs/celltypes/workgroups/rnaseqanalysis/dyuan/custom_packages/mpi_tc/archive_versions/v3_githubVersion/test_70k_worked/out/clustering_results.pkl'
with open(clusters_pth, 'rb') as f:
    clusters = pickle.load(f)

# convert to a dictionary of cluster id to cell indices
obs_by_cluster = defaultdict(lambda: [])
for i, cell_ids in enumerate(clusters):
    obs_by_cluster[i] = cell_ids

# subset the adata to include only the first two clusters for testing
adata = adata[list(obs_by_cluster[0]) + list(obs_by_cluster[1]), :].copy()
# create the new obs_by_cluster of cluster id to the cell indices in the subsetted adata
obs_by_cluster_subset = {}
obs_by_cluster_subset[0] = [i for i in range(len(obs_by_cluster[0]))]
obs_by_cluster_subset[1] = [i + len(obs_by_cluster[0]) for i in range(len(obs_by_cluster[1]))]

# returns a set of markers (20 up regulated and 20 down regulated)
degs = pairwise_degs(
        adata,
        obs_by_cluster_subset,
        thresholds,
        n_markers=20,
        de_method='ebayes',
        n_jobs=30
        )