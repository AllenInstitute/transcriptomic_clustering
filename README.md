# transcriptomic_clustering

`transcriptomic_clustering` is a Python toolkit for building and evaluating
clustering workflows for transcriptomic data. It is designed primarily for
single-cell research and works with [AnnData](https://anndata.readthedocs.io/)
objects and the [Scanpy](https://scanpy.readthedocs.io/) ecosystem.

The package provides tools for:

- normalization and highly variable gene selection;
- dimensionality reduction and projection;
- Louvain, Leiden, and Phenograph clustering;
- one-step and iterative clustering workflows;
- differential-expression analysis and cluster merging; and
- hierarchical sorting of cluster summaries.

> **Research software and reproducibility**
>
> This repository supports research workflows and is updated occasionally,
> without a fixed release or support schedule. For reproducible analyses, use a
> pinned commit or release and record your environment and parameters.
> Community questions, bug reports, and pull requests are welcome.

## Documentation and examples

- [Installation guide](docs/install.rst)
- [Example notebooks](docs/notebooks/)
- [Contributing guide](CONTRIBUTING.md)
- [Iterative clustering command-line script](scripts/run_iter_clust.py)

The notebooks include a general
[demonstration](docs/notebooks/demo_060121.ipynb) and an
[iterative clustering example](docs/notebooks/iterative_clustering.ipynb).

## Quick start

The supported environment is Python 3.8 on Linux. Clone the repository, create
an isolated environment, install the requirements, and install the package in
editable mode:

```bash
git clone https://github.com/AllenInstitute/transcriptomic_clustering.git
cd transcriptomic_clustering

python3.8 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Verify the installation:

```python
import transcriptomic_clustering

print(transcriptomic_clustering.__name__)
```

See the [installation guide](docs/install.rst) for additional installation
options and environment-management guidance.

## Main tasks and APIs

Functions are available from the modules shown below.

| Task | Public function(s) | Module |
| --- | --- | --- |
| Normalize expression data | `normalize` | `transcriptomic_clustering.normalization` |
| Select highly variable genes | `highly_variable_genes` | `transcriptomic_clustering.highly_variable_genes` |
| Reduce dimensions | `pca` | `transcriptomic_clustering.dimension_reduction` |
| Project data | `project`, `latent_project` | `transcriptomic_clustering.projection` |
| Run Louvain, Leiden, or Phenograph clustering | `cluster_louvain`, `cluster_louvain_phenograph`, `get_taynaud_louvain`, `get_vtraag_leiden` | `transcriptomic_clustering.clustering` |
| Run a single clustering pass | `onestep_clust` | `transcriptomic_clustering.onestep_clustering` |
| Run iterative clustering | `iter_clust` | `transcriptomic_clustering.iterative_clustering` |
| Calculate differential expression | `de_pairs_chisq`, `de_pairs_ebayes` | `transcriptomic_clustering.diff_expression`, `transcriptomic_clustering.de_ebayes` |
| Merge clusters | `merge_clusters`, `merge_small_clusters`, `merge_clusters_by_de` | `transcriptomic_clustering.merging` |
| Hierarchically sort cluster means | `hclust` | `transcriptomic_clustering.hierarchical_sorting` |

Refer to the function docstrings and
[example notebooks](docs/notebooks/) for inputs, outputs, and workflow context.

## Development and testing

After completing the [quick start](#quick-start), install the test dependencies
and run the existing test suite:

```bash
python -m pip install -r test_requirements.txt
pytest tests
```

Please add tests and documentation for new behavior, and follow the workflow in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Support

This project receives occasional updates with no fixed schedule. Search the
[existing issues](https://github.com/AllenInstitute/transcriptomic_clustering/issues)
before opening a new bug report or question. Community involvement through
issues and pull requests is encouraged.

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for
the development workflow, testing expectations, style guidance, and
contribution terms.

## License

This project is distributed under the
[Allen Institute Software License](LICENSE). Redistribution and noncommercial
use are permitted under its terms. Commercial redistribution or use requires
written permission from the Allen Institute as specified in the license;
contact `terms@alleninstitute.org` for commercial licensing opportunities.
