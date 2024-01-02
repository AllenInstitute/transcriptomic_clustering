# -*- coding: utf-8 -*-

"""Top-level package for transcriptomic_clustering."""

import os
from logging.config import fileConfig


version_path = os.path.join(os.path.dirname(__file__), "VERSION.txt")
with open(version_path, "r") as version_file:
    __version__ = version_file.read().strip()


fileConfig(os.path.join(
    os.path.dirname(__file__), 
    'logging_config.ini')
)

# paths
from .utils.memory import memory
from .normalization import normalize
from .highly_variable_genes import highly_variable_genes
from .means_vars_genes import get_means_vars_genes
from .dimension_reduction import pca
from .projection import project, latent_project
from .clustering import cluster_louvain, cluster_louvain_phenograph
from .filter_known_modes import filter_known_modes
from .hierarchical_sorting import hclust
from .cluster_means import get_cluster_means
from .merging import merge_clusters
from .diff_expression import de_pairs_chisq, vec_chisq_test
from .de_ebayes import de_pairs_ebayes
from .merging import merge_clusters
