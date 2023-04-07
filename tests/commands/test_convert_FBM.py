from pathlib import Path

import pytest
from click.testing import CliRunner

import numpy as np
import anndata as ad

from transcriptomic_clustering.commands.convert_FBM import convert_FBM_cmd, DTYPE_DICT

@pytest.mark.parametrize("normalize,in_dtype,out_dtype",
    [
        (False, "double64", "double64"),
        (False, "float32", "float32"),
        (False, "int32", "int32"),
        (False, "int16", "int16"),
        (False, "double64", "float32"),
        (False, "int32", "float32"),
        (False, "int16", "int32"),
        (True, "double64", "float32"),
        (True, "int16", "float32")
    ]
)
def test_convert_FBM(tmp_path, normalize, in_dtype, out_dtype):
    nobs = 10000
    nvar = 3000
    ndata = nobs * nvar

    test_data = np.linspace(0, ndata, ndata, endpoint=False, dtype=DTYPE_DICT[in_dtype])
    test_data = test_data.reshape(nobs, nvar)
    test_data_R = test_data.T

    fbm_path = tmp_path / "fbm.bk"
    fbm = np.memmap(
        fbm_path, dtype=test_data.dtype, mode='w+', shape=(nvar, nobs), order="F"
    )
    fbm[:] = test_data_R[:]

    genes = [f"Gene_{i}" for i in range(nvar)]
    cells = [f"Cell_{i}" for i in range(nobs)]
    gene_csv_path = tmp_path / 'genes.csv'
    cells_csv_path = tmp_path / 'cells.csv'

    with open(gene_csv_path, 'w') as f:
        f.write('x\n' + '\n'.join(genes))
    with open(cells_csv_path, 'w') as f:
        f.write('x\n' + '\n'.join(cells))
    
    cmd_kwargs = {
        "-p": in_dtype,
        "-c": "1000",
        "-d": out_dtype,
    }

    cmd_args = []
    cmd_args.append(str(fbm_path))
    cmd_args.append(str(gene_csv_path))
    cmd_args.append(str(cells_csv_path))
    for k, v in cmd_kwargs.items():
        cmd_args.append(k)
        cmd_args.append(v)
    if normalize:
        cmd_args.append("-n")

    runner = CliRunner()
    result = runner.invoke(convert_FBM_cmd, cmd_args)
    assert result.exit_code == 0
    result_lines = result.output.split("\n")
    out_path = result_lines[-2]

    if normalize:
        expected_out_path = str(tmp_path / 'fbm_normalized.h5ad')
    else:
        expected_out_path = str(tmp_path / 'fbm.h5ad')
    assert out_path == expected_out_path

    obt_ad = ad.read_h5ad(out_path)
    assert obt_ad.n_obs == nobs
    assert obt_ad.n_vars == nvar
    assert obt_ad.X.dtype == DTYPE_DICT[out_dtype]
    if normalize:
        np.testing.assert_allclose(np.expm1(obt_ad.X[0,:]).sum(), 1e6)
    else:
        np.testing.assert_allclose(obt_ad.X, test_data)
