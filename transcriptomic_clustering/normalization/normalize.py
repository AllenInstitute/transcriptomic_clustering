from typing import Dict, Optional, List, Any

import logging
import copy as cp

from argschema.argschema_parser import ArgSchemaParser
from transcriptomic_clustering.normalization._schemas import (
    InputParameters, OutputParameters)

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix


def normalize_cell_expresions(cell_expressions: ad.AnnData):
    """
        Compute the normalization of cell expressions

            (1) compute cpm (counts per million normalization): scanpy.pp.normalize_total,
                which computes "single_expression/sum(all_expressions_per_cell)*1e6"
            (2) compute log2: scipy.sparse.csr_matrix.log1p(x)/np.log(2.0),
                which computes log(x+1)/log(2)

        Parameters
        ----------
        cell_expressions: AnnData  

        Returns
        -------
        normalization result: AnnData

    """

    # cpm
    sc.pp.normalize_total(cell_expressions, target_sum=1e6, inplace=True)

    # log
    cell_expressions.X = csr_matrix.log1p(cell_expressions.X)/np.log(2.0)

    return cell_expressions


def main():
    parser = ArgSchemaParser(
        schema_type=InputParameters,
        output_schema_type=OutputParameters
    )

    args = cp.deepcopy(parser.args)
    logging.getLogger().setLevel(args.pop("log_level"))

    adata = sc.read_h5ad(args["cell_expression_path"])
    output = normalize_cell_expresions(adata)

    parser.output(output)


if __name__ == "__main__":
    main()
