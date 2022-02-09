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
from tqdm import tqdm


def copy_anndata_without_X(
        adata: AnnData,
        filename: Union[Path, str]
) -> AnnData:


    """
    Create a new file-backed adata that that has annotations from the source adata object, but with empty X

    Parameters
    ----------
    adata: AnnData object
    filename: name of h5ad file to create

    Returns
    -------
    An :class:`AnnData` object
    """

    if os.path.isfile(filename):
        raise ValueError(f"File {filename} already exists. "
                         f"Cannot back to the existing file.")

    f = h5py.File(filename, "w")
    if adata.X.format_str != 'csr':
        raise TypeError("Writing to backed file supports only CSR format")

    ad._io.h5ad.write_elem(f, "X", csr_matrix((0, adata.n_vars), dtype='float32'))
    ad._io.h5ad.write_elem(f, "obs", adata.obs)
    ad._io.h5ad.write_elem(f, "var", adata.var)
    f.close()

    adata = sc.read_h5ad(filename, backed='r+')
    return adata


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

    counts_per_cell = adata.X.sum(1)
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
    process_memory_est = adata.n_obs * adata.n_vars * adata.X.dtype.itemsize/(1024**3)
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

    if copy_to:
        adata_output = copy_anndata_without_X(adata, copy_to)
    else:
        raise AttributeError(
            "Missing required `copy_to` argument for the file-backed processing"
        )

    for chunk, start, end in tqdm( adata.chunked_X(chunk_size), desc="processing", total=nchunks ):
        sys.stdout.flush()
        counts_per_cell = chunk.sum(1)
        chunk = sc.pp._normalization._normalize_data(chunk, counts_per_cell, after=1e6)
        chunk = sc.pp.log1p(chunk)
        adata_output.X.append(chunk)


    return adata_output

