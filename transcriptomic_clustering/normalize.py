from typing import Dict, Optional, List, Any

import logging
import copy as cp
import sys
import h5py
from os import PathLike
import numpy as np
import pandas as pd
from anndata import AnnData
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix, issparse, vstack


def normalize(cell_expressions: ad.AnnData):
    """
        Compute the normalization of cell expressions

            (1) compute cpm (counts per million): The counts per gene were normalized to CPM 
                by dividing it by the total number of mapped reads per sample and multiplying by 1,000,000
            (2) compute log2: computes log2(x+1)

        Parameters
        ----------
        cell_expressions: AnnData format, both dense matrix and sparse matrix (csr_matrix) are supported
            The annotated data matrix of shape n_obs × n_vars.
            Rows correspond to cells and columns to genes

        Returns
        -------
        normalization result: log(cpm+1) in AnnData format

    """

    # cpm
    sc.pp.normalize_total(cell_expressions, target_sum=1e6, inplace=True)

    # log
    if issparse(cell_expressions.X):
        if cell_expressions.X.getformat() == 'csr':
            sc.pp.log1p(cell_expressions)
        else:
            raise ValueError("Unsupported format for cell_expression matrix. Must be in CSR or dense format")
    else:
        cell_expressions.X = np.log1p(cell_expressions.X)

    return cell_expressions

#@profile
def log1p_of_cpm(
        adata: AnnData,
        inplace: bool = True,
        filename: PathLike = None,
        chunk_size: int = None,
    ):

    if adata.isbacked:
        if inplace:
            raise NotImplementedError(
                "Cannot write to same backed file. Use inplace=False and"
                "provide a filename where results will be saved"
            )
        else:
            if filename:
                f = h5py.File(filename, "w")
                ad._io.h5ad.write_attribute(f, "X", csr_matrix((0, adata.n_vars)))
                dataset = ad._core.sparse_dataset.SparseDataset(f["X"])
                ad._io.h5ad.write_attribute(f, "obs", adata.obs)
                ad._io.h5ad.write_attribute(f, "var", adata.var)

                print(f"Will save output to {filename}")
                n_chunks=10
                print(f"Processing in {n_chunks} chunks with chunk_size: {chunk_size}")
                for chunk, start, end in adata.chunked_X(chunk_size):
                    counts_per_cell = chunk.sum(1)
                    print(".", end="")
                    sys.stdout.flush()
                    chunk = sc.pp._normalization._normalize_data(chunk, counts_per_cell, after=1e6)
                    chunk = sc.pp.log1p(chunk, base=2)
                    dataset.append(chunk)

                print("finished!")
                f.close()
                adata = sc.read_h5ad(filename, backed='r')

            else:
                raise AttributeError(
                    "Missing `filename` argument for backed file with inplace=False"
                )
        print(f"Processing with chunk_size: {chunk_size}")

    else:
        print("processing without chunking")
        if inplace:
            print("processing in memory in place")
            sc.pp.normalize_total(adata, target_sum=1e6)
            sc.pp.log1p(adata, base=2)

        else:
            print("processing in memory making a copy")
            adata.X = sc.pp.normalize_total(adata, target_sum=1e6, inplace=False)['X']
            adata = sc.pp.log1p(adata, base=2, copy=True)

    return adata if not inplace else None
