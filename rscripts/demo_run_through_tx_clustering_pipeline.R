## ----Load libraries, messages = F, include=FALSE------------------------------
library(dendextend)
library(matrixStats)
library(Matrix)
library(scrattch.hicat)

# devtools::install_github("AllenInstitute/scrattch.io")
library(scrattch.io)
library(rhdf5)

## ----tasic2016data------
library(tasic2016data)

## ----Set up anno--------------------------------------------------------------
# Load sample annotations (anno)
anno <- tasic_2016_anno

# Make a data.frame of unique cluster id, type, color, and broad type
ref.cl.df <- as.data.frame(unique(anno[,c("primary_type_id", "primary_type_label", "primary_type_color", "broad_type")]))

#standardize cluster annoation with cluster_id, cluster_label and cluster_color. These are the required fields to visualize clusters properly.
colnames(ref.cl.df)[1:3] <- c("cluster_id", "cluster_label", "cluster_color")

# Sort by cluster_id
ref.cl.df <- ref.cl.df[order(ref.cl.df$cluster_id),]
row.names(ref.cl.df) <- ref.cl.df$cluster_id

ref.cl <- setNames(factor(anno$primary_type_id), anno$sample_name)

## ----Normalize data-----------------------------------------------------------
ptm <- proc.time()
norm.dat <- Matrix(cpm(tasic_2016_counts), sparse = TRUE)
norm.dat@x <- log2(norm.dat@x+1)
proc.time() - ptm

save_sparse_matrix_h5(norm.dat, "normalize_result.h5")

## ----Filter samples-----------------------------------------------------------
select.cells <- with(anno, sample_name[primary_type_label!="unclassified" & grepl("Igtp|Ndnf|Vip|Sncg|Smad3",primary_type_label)])

length(select.cells)

## ----Set Params---------------------------------------------------------------
de.param <- de_param(padj.th     = 0.05,
                     lfc.th      = 1,
                     low.th      = 1,
                     q1.th       = 0.5,
                     q.diff.th   = 0.7,
                     de.score.th = 40)

## ----Run iter_clust, message=FALSE, warning=FALSE, echo=TRUE------------------
#ptm <- proc.time()
#onestep.result <- onestep_clust(norm.dat,
#                               select.cells = select.cells,
#                               dim.method = "WGCNA",
#                               de.param = de_param(de.score.th=500))
#proc.time() - ptm


## select high variance genes
method = c("louvain","leiden","ward.D", "kmeans")
dim.method = c("pca","WGCNA")
merge.type = c("undirectional", "directional")
maxGenes = 3000
max.cl.size = 300
max.dim = 20
verbose = FALSE

sampled.cells = select.cells

select.genes = row.names(norm.dat)[which(Matrix::rowSums(norm.dat[,select.cells] > de.param$low.th) >= de.param$min.cells)]

counts = norm.dat[,sampled.cells]
counts@x = 2^(counts@x) - 1
dim.method="WGCNA"

ptm <- proc.time()
vg = find_vg(as.matrix(counts[select.genes,sampled.cells]),plot_file=NULL)
select.genes = as.character(vg[which(vg$loess.padj < 1),"gene"])
select.genes = head(select.genes[order(vg[select.genes, "loess.padj"],-vg[select.genes, "z"])],maxGenes)
proc.time() - ptm

length(select.genes)

## dimension reduction
ptm <- proc.time()
rd.dat = rd_WGCNA(norm.dat, select.genes=select.genes, select.cells=select.cells, sampled.cells=sampled.cells, de.param=de.param, max.mod=max.dim, max.cl.size=max.cl.size)$rd.dat
proc.time() - ptm


##
minModuleSize=10
cutHeight=0.99
type="unsigned"
softPower=4
rm.gene.mod=NULL
rm.eigen=NULL

dat <- as.matrix(norm.dat[select.genes, sampled.cells])
adj <- WGCNA::adjacency(t(dat), power = softPower, type = type)
adj[is.na(adj)] <- 0

TOM <- WGCNA::TOMsimilarity(adj, TOMType = type, verbose = 0)

dissTOM <- as.matrix(1 - TOM)
rownames(dissTOM) <- rownames(dat)
colnames(dissTOM) <- rownames(dat)

geneTree <- hclust(as.dist(dissTOM),
                     method = "average")
  
dynamicMods <- dynamicTreeCut::cutreeDynamic(dendro = geneTree,
                                           distM = dissTOM,
                                           cutHeight = cutHeight,
                                           deepSplit = 2,
                                           pamRespectsDendro = FALSE,
                                           minClusterSize = minModuleSize)

gene.mod <- split(row.names(dissTOM), dynamicMods)
gene.mod <- gene.mod[setdiff(names(gene.mod), "0")]

View(gene.mod)

## filter gene mod
ptm <- proc.time()
gm <- filter_gene_mod(norm.dat,
                        select.cells,
                        gene.mod,
                        minModuleSize = minModuleSize,
                        rm.eigen = rm.eigen)
proc.time() - ptm

View(gm$eigen)


## clustering
max.cl = ncol(rd.dat)*2 + 1
k.nn = 15
k = pmin(k.nn, round(nrow(rd.dat)/2))
ptm <- proc.time()
tmp = jaccard_louvain(rd.dat, k)
proc.time() - ptm
cl = tmp$cl
if(length(unique(cl))>max.cl){
  tmp.means =do.call("cbind",tapply(names(cl),cl, function(x){
    colMeans(rd.dat[x,,drop=F])
  },simplify=F))
  tmp.hc = hclust(dist(t(tmp.means)), method="average")
  tmp.cl= cutree(tmp.hc, pmin(max.cl, length(unique(cl))))
  cl = setNames(tmp.cl[as.character(cl)], names(cl))
}

# merge
rd.dat.t = t(rd.dat)
ptm <- proc.time()
merge.result=merge_cl(norm.dat, cl=cl, rd.dat.t=rd.dat.t, merge.type=merge.type, de.param=de.param, max.cl.size=max.cl.size,verbose=verbose)
proc.time() - ptm

# hierarchical sorting
sc = merge.result$sc
#print(sc)
cl = merge.result$cl
if(length(unique(cl))>1){
    #if(verbose){
    #  cat("Expand",prefix, "\n")
    #  cl.size=table(cl)
    #  print(cl.size)
    #  save(cl, file=paste0(prefix, ".cl.rda"))
    #}
    de.genes = merge.result$de.genes
    markers= merge.result$markers
    cl.dat = get_cl_means(norm.dat[markers,], cl[sample_cells(cl, max.cl.size)])
    cl.hc = hclust(dist(t(cl.dat)),method="average")
    cl = setNames(factor(as.character(cl), levels= colnames(cl.dat)[cl.hc$order]), names(cl))
    #if(verbose & !is.null(prefix)){
    #  tmp=display_cl(cl, norm.dat, prefix=prefix, markers=markers, max.cl.size=max.cl.size)
    #}
    levels(cl) = 1:length(levels(cl))
    result=list(cl=cl, markers=markers)
}

## ---- fig.height=7, fig.width=7-----------------------------------------------
display.result = display_cl(onestep.result$cl, norm.dat, plot=TRUE, de.param=de.param)


## ---- message=FALSE, warning=FALSE, results="hide"----------------------------
ptm <- proc.time()
WGCNA.clust.result <- iter_clust(norm.dat,
                               select.cells = select.cells,
                               dim.method = "WGCNA",
                               de.param = de.param,
                               result=onestep.result)
proc.time() - ptm


## ---- message=FALSE, warning=FALSE, results="hide"----------------------------
gene.counts <- colSums(norm.dat > 0)
rm.eigen <- matrix(log2(gene.counts), ncol = 1)
row.names(rm.eigen) <- names(gene.counts)
colnames(rm.eigen) <- "log2GeneCounts"

## ----Merge clusters, message=FALSE, warning=FALSE, result="hide"--------------
ptm <- proc.time()
WGCNA.merge.result <- merge_cl(norm.dat,
                         cl = WGCNA.clust.result$cl,
                         rd.dat = t(norm.dat[WGCNA.clust.result$markers, select.cells]),
                         de.param = de.param)
proc.time() - ptm


## ----Compare to benchmark, echo=FALSE, fig.height = 5, fig.width = 6.5--------
compare.result <- compare_annotate(WGCNA.merge.result$cl, ref.cl, ref.cl.df)
compare.result$g
cl <- compare.result$cl
cl.df <- compare.result$cl.df

## ---- fig.height=7, fig.width=7-----------------------------------------------
display.result = display_cl(cl, norm.dat, plot=TRUE, de.param=de.param, min.sep=4, n.markers=20)
de.genes= display.result$de.genes

## ----Drop cluster levels------------------------------------------------------
cl.clean <- droplevels(cl)

## ----Build dendrogram, warning=FALSE, message=FALSE, results="hide", fig.keep="all", fig.height = 4.5, fig.width = 7----
select.markers = select_markers(norm.dat, cl.clean, de.genes=de.genes,n.markers=50)$markers
cl.med <- get_cl_medians(norm.dat[select.markers,], cl)
##The prefered order for the leaf nodes.
l.rank <- setNames(1:nrow(cl.df), row.names(cl.df))
##Color of the leaf nodes.
l.color <- setNames(as.character(cl.df$cluster_color), row.names(cl.df))
dend.result <- build_dend(cl.med[,levels(cl.clean)],
                          l.rank,
                          l.color,
                          nboot = 100)
dend <- dend.result$dend
###attach cluster labels to the leafs of the tree
dend.labeled = dend
labels(dend.labeled) <- cl.df[labels(dend), "cluster_label"]
plot(dend.labeled)

## ----Reorder dendrogram-------------------------------------------------------
cl.clean <- setNames(factor(as.character(cl.clean), levels = labels(dend)), names(cl.clean))
cl.df.clean <- cl.df[levels(cl.clean),]
