from typing import Dict, Optional, List, Any

import os
import math
import sys
import h5py
from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd
from anndata import AnnData
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix, issparse, vstack
import warnings
import transcriptomic_clustering as tc
from transcriptomic_clustering.iter_writer import AnnDataIterWriter
from tqdm import tqdm


def normalize(
    adata:AnnData,
    inplace: bool = False,
    chunk_size: int = None,
    copy_to: Union[Path, str] = None
) -> AnnData:
    """
    Normalize data by:

            (1) normalize counts per cell to sum to million (CPM)  per sample and multiplying by 1,000,000
            (2) compute log1p of CPM data

    Handles both in-memory and file-backed data.

    Parameters
    ----------
    adata: AnnData object
    inplace: whether to modify the existing object and return a new object
    chunk_size: number of observations to process in a single chunk.
    copy_to: filename of where to back returned data

    Returns
    -------
    Returns or updates `adata`, depending on `inplace`.
    """

    if adata.isbacked:
        if inplace:
            raise NotImplementedError(
                "Inplace update is not supported for file-backed data. "
                "Use inplace=False and provide a backing filename with `copy_to`"
            )

        adata_output = normalize_backed(adata, chunk_size, copy_to)

    else:
        if chunk_size:
            warnings.warn("In memory processing does not support chunking. "
                          "Ignoring `chunk_size` argument.")
        if copy_to:
            warnings.warn("In memory processing does not support backing data to file. "
                          "Ignoring `copy_to` argument.")
        if inplace:
            warnings.warn("Modifying data in place.")

        adata_output = normalize_inmemory(adata, inplace)

    return adata_output


def normalize_inmemory(
    adata: AnnData,
    inplace: bool = True
) -> AnnData:

    """
    Normalize data located in memory
    See description of normalize() for details
    """
    adata = adata.copy() if not inplace else adata

    counts_per_cell = np.ravel(adata.X.sum(1))
    adata.X = sc.pp._normalization._normalize_data(adata.X, counts_per_cell, after=1e6)
    adata.X = sc.pp.log1p(adata.X)

    return adata


def normalize_backed(
    adata: AnnData,
    chunk_size: int = None,
    copy_to: Union[Path,str] = None
) -> AnnData:
    """
    Normalize file-backed data in chunks
    See description of normalize() for details
    """
    if not adata.is_view:  # .X on view will try to load entire X into memory
        itemsize = adata.X.dtype.itemsize
    else:
        itemsize = np.dtype(np.float64).itemsize
    process_memory_est = adata.n_obs * adata.n_vars * itemsize/(1024**3)
    output_memory_est = 0.1 * process_memory_est

    estimated_chunk_size = tc.memory.estimate_chunk_size(
        adata,
        process_memory=process_memory_est,
        output_memory=output_memory_est,
        percent_allowed=100,
        process_name='normalize'
    )

    if chunk_size:
        if not(chunk_size >= 1 and isinstance(chunk_size, int)):
            raise ValueError("chunk_size argument must be a positive integer")

        if estimated_chunk_size < chunk_size:
           warnings.warn(f"Selected chunk_size: {chunk_size} is larger than "
                         f"the estimated chunk_size {estimated_chunk_size}. "
                         f"Using chunk_size larger than recommended may result in MemoryError")
    else:
        chunk_size = estimated_chunk_size

    nchunks = math.ceil(adata.n_obs/chunk_size)

    if not copy_to:
        raise AttributeError(
            "Missing required `copy_to` argument for the file-backed processing"
        )
    writer = None
    for chunk, start, end in tqdm( adata.chunked_X(chunk_size), desc="processing", total=nchunks ):
        sys.stdout.flush()
        counts_per_cell = np.ravel(chunk.sum(1))
        chunk = sc.pp._normalization._normalize_data(chunk, counts_per_cell, after=1e6)
        chunk = sc.pp.log1p(chunk)
        if writer is None:
            writer = AnnDataIterWriter(copy_to, chunk, adata.obs, adata.var)
        else:
            writer.add_chunk(chunk)

    if writer is None:
        raise ValueError("Adata was empty!")
    return writer.adata
