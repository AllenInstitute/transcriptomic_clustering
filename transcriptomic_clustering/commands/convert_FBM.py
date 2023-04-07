from typing import Optional, Literal
import sys
from pathlib import Path
import time
import logging
from enum import Enum

import click

import numpy as np
import pandas as pd
from scipy import sparse
import h5py
from tqdm import tqdm

import anndata as ad
import scanpy as sc
import transcriptomic_clustering as tc
from transcriptomic_clustering.iter_writer import AnnDataIterWriter


logger = logging.getLogger(__name__)


DTYPE_DICT = {
    'double64': np.float64,
    'float32': np.float32,
    'int32': np.int32,
    'int16': np.int16,
}
DTYPE_NAMES = ["double64", "float32", "int32", "int16"]
DTYPE_NAMES_TYPE = Literal["double64", "float32", "int32", "int16"]


def chunked_fbm(fbm: np.memmap, chunk_size: int = 1000):
    """
    Return an iterator over the rows of the fbm.
    Parameters
    ----------
    fbm:
        numpy memmap matrix
    chunk_size
        Row size of a single chunk.
    """
    start = 0
    n = fbm.shape[0]
    for _ in range(int(n // chunk_size)):
        end = start + chunk_size
        yield (fbm[start:end], start, end)
        start = end
    if start < n:
        yield (fbm[start:n], start, n)


def convert_FBM(
        fbm_path: Path,
        gene_csv_path: Path,
        cell_csv_path: Path,
        fbm_dtype: DTYPE_NAMES_TYPE = "double64",
        out_ad_path: Optional[Path] = None,
        chunk_size: int = 5000,
        out_dtype: DTYPE_NAMES_TYPE = None,
        as_sparse: bool = False,
        normalize: bool = True,
        target_sum: Optional[float] = 1e6,
):
    """
    Creates a dense filebacked AnnData h5py object from an FBM file
    and optionally normalizes it

    Parameters
    ----------
    fbm_path:
        path to filebackedmatrix (see R's filebacked.big.matrix)
    gene_csv_path:
        path to csv file containing a single column of gene names with header
    cell_csv_path:
        path to csv file containing a single column of cell names with header
    fbm_dtype:
        precision of fbm (float: 4 bytes, double: 8 bytes)
    out_ad_path:
        path to save filebacked AnnData at
    chunk_size:
        number of rows to process at a time
    out_dtype:
        precision to save AnnData X matrix as (float32: 4 bytes, double: 8 bytes)
    normalize:
        if true, will normalize counts to <target_sum> or median cell count, and then
        calculate log1p of data
    target_sum:
        counts to normalize data to.

    Returns
    -------
    path to new anndata file
    """
    gene_df = pd.read_csv(gene_csv_path, index_col=0, header=0)
    cell_df = pd.read_csv(cell_csv_path, index_col=0, header=0)
    nobs = len(cell_df)
    nvar = len(gene_df)

    fbm_dtype = DTYPE_DICT[fbm_dtype]

    # For sanity checking np.memmap
    estimated_FBM_GB = np.dtype(fbm_dtype).itemsize * nobs * nvar / (1024 ** 3)
    logger.info(
        f"Estimated size of {str(fbm_path)} on disk: {estimated_FBM_GB} GB"
    )

    # memmap matrix
    fbm_mat = np.memmap(
        fbm_path,
        mode="r",
        shape=(nvar, nobs),  # row: gene, col: cell
        dtype=fbm_dtype,
        order="F",  # R is column Major
    ).T  # to get to row: cell, col: gene

    # Set out_ad_path
    if out_ad_path is None:
        ad_name = fbm_path.stem
        if normalize:
            ad_name += "_normalized"
        out_ad_path = fbm_path.with_name(f"{ad_name}.h5ad")

    # Check if needs to be cast
    cast = False
    if out_dtype is not None and out_dtype != fbm_dtype:
        out_dtype = DTYPE_DICT[out_dtype]
        cast = True

    # Write to AnnData
    tic = time.perf_counter()
    nchunks = int(nobs // chunk_size)
    ad_writer = None
    for chunk, start, end in tqdm(
            chunked_fbm(fbm_mat, chunk_size=chunk_size), total=nchunks):

        if cast:
            chunk = chunk.astype(out_dtype, casting="same_kind", copy=False)
        if normalize:
            counts_per_cell = np.ravel(chunk.sum(1))
            chunk = sc.pp._normalization._normalize_data(
                chunk, counts_per_cell, after=target_sum
            )
            chunk = sc.pp.log1p(chunk)
        if as_sparse:
            chunk = sparse.csr_matrix(chunk)

        if ad_writer is None:
            ad_writer = AnnDataIterWriter(out_ad_path, chunk, cell_df, gene_df, dtype=out_dtype)
        else:
            ad_writer.add_chunk(chunk)

    toc = time.perf_counter()
    logger.info(f"Elapsed Time: {toc - tic}")
    logger.info(f"Finished creating {out_ad_path}")
    return out_ad_path


@click.command()
@click.argument(
    "fbm_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "gene_csv_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "cell_csv_path",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "-p", "--fbm_dtype", "fbm_dtype",
    help="precision of data in FBM matrix: int32: 4 bytes, float32: 4 bytes, double64: 8 bytes",
    type=click.Choice(DTYPE_NAMES, case_sensitive=False)
)
@click.option(
    "-o", "--output_path", "out_ad_path",
    type=click.Path(exists=False, path_type=Path),
)
@click.option(
    "-d", "--out_dtype", "out_dtype",
    help="precision of data to output: int32: 4 bytes, float32: 4 bytes, double64: 8 bytes",
    type=click.Choice(DTYPE_NAMES, case_sensitive=False)
)
@click.option(
    "-s", "--sparse", "as_sparse",
    help="Save as sparse matrix",
    is_flag=True
)
@click.option(
    "-c", "--chunk_size", "chunk_size",
    help="size of chunk to process in rows (will load all columns)",
    type=int
)
@click.option(
    "-n", "--normalize", "normalize",
    help="Normalize data during copy",
    is_flag=True
)
@click.option(
    "-t", "--target_sum", "target_sum",
    help="Normalize cells to target sum count",
    type=int,
    default=1e6
)
def convert_FBM_cmd(*args, **kwargs):
    out_ad_path = convert_FBM(*args, **kwargs)
    click.echo(str(out_ad_path))


if __name__ == "__main__":
    convert_FBM_cmd()
