from typing import Dict, Optional, List, Any

import logging
import copy as cp

from transcriptomic_clustering.normalization._schemas import (
    InputParameters, OutputParameters)

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix, issparse


def normalize_cell_expresions(cell_expressions: ad.AnnData):
    """
        Compute the normalization of cell expressions

            (1) compute cpm: count up the total reads in a sample and divide that number by 1,000,000
            (2) compute log2: computes log(x+1)/log(2)

        Parameters
        ----------
        cell_expressions: input in AnnData format with sparse matrix (csr_matrix) 

        Returns
        -------
        normalization result: output in AnnData format

    """

    # cpm
    sc.pp.normalize_total(cell_expressions, target_sum=1e6, inplace=True)

    # log
    if(issparse(cell_expressions.X)):
        cell_expressions.X = csr_matrix.log1p(cell_expressions.X)/np.log(2.0)
    else:
        cell_expressions.X = np.log2(cell_expressions.X)

    return cell_expressions

