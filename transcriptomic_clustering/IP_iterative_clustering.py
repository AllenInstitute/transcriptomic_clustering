# Transcriptomic Clustering

import scanpy as sc
import transcriptomic_clustering as tc

SPLIT_SIZE = 4

# (CK) A single step of clustering. Takes a list of samples and an AnnData object. Returns a dictionary
# where cluster indexes are keys and lists of samples are values.
def onestep_clust(samples, adata,
                   pc_filtering=False,
                   thresholds=None):

   normalized_adata = adata[samples, :]

   #Highly Variant Genes
   means, variances, gene_mask = tc.get_means_vars_genes(adata=normalized_adata)
   tc.highly_variable_genes(adata=normalized_adata,
                         means=means, variances=variances,
                         gene_mask=gene_mask, max_genes=3000)
   print(normalized_adata)
   #PCA
   (components, explained_variance_ratio, explained_variance, means) =  \
      tc.pca(normalized_adata, n_comps=25, cell_select=1000, use_highly_variable=True, svd_solver='arpack')
   print(components.shape)
   #Filter Known Modes
   if pc_filtering:
      known_modes = components[[24]] # select last component as a known mode as an example
      components = tc.filter_known_modes(components, known_modes)
      print(components.shape)
   else:
      print("no PC filtering")
   #Projection
   projected_adata = tc.project(normalized_adata, components, means)
   print(projected_adata)
   #Louvain Clustering
   cluster_by_obs, obs_by_cluster, graph, qc = tc.cluster_louvain(projected_adata, 10, n_jobs=8,random_seed=12341)
   cluster_sizes_before_merging = {k: len(v) for k, v in obs_by_cluster.items()}
   #Merging
   if (thresholds==None):
      thresholds = {
         'q1_thresh': 0.5,
         'q2_thresh': None,
         'cluster_size_thresh': 15,
         'qdiff_thresh': 0.7,
         'padj_thresh': 0.05,
         'lfc_thresh': 1.0,
         'score_thresh': 200,
         'low_thresh': 1
      }
   cluster_assignments_after_merging = tc.merge_clusters(
       adata_norm=normalized_adata,
       adata_reduced=projected_adata,
       cluster_assignments=obs_by_cluster,
       cluster_by_obs=cluster_by_obs,
       thresholds=thresholds,
       de_method='ebayes'
   )
   #Hierarchical Sorting
   #import numpy as np
   #cluster_by_obs_after_merging = np.zeros(len(cluster_by_obs), dtype=int)
   #for cluster, obs in results.items():
   #   cluster_by_obs_after_merging[obs] = cluster
   return cluster_assignments_after_merging


# (Santino) Builds a cluster dictionary from a list of lists of samples, each represents a cluster.
def build_cluster_dict(clusters):
    output = {}
    for i in range(len(clusters)):
        output[i + 1] = clusters[i]
    return output

# (Santino) Iteratively applies one step of clustering to resultant clusters until convergence. Takes a list of samples,
# an AnnData object, and a list of lists.
def iter_cluster(samples, adata, clusters):
    if len(samples) > 0:
        if len(samples) <= MIN_SAMPLE_SIZE:
            clusters.append(samples)
        else:
            next = onestep_clust(samples, adata)
            if len(next) == 1:
                clusters.append(samples)
            else:
                for cluster in next:
                    iter_cluster(next[cluster], adata, clusters)

def main():
    tasic_adata = sc.read_h5ad('/home/changkyul/CK/Iter_Clust_R2Python/transcriptomic_clustering/docs/notebooks/data/tasic2016counts_sparse.h5ad')
    normalized_adata = tc.normalize(tasic_adata)
    clusters = []
    initial_samples = normalized_adata.obs.index.values
    iter_cluster(initial_samples, normalized_adata, clusters)
    output = build_cluster_dict(clusters)
    print(output)


if __name__ == "__main__":
    main()
