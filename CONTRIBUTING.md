Contributing
============

Thank you for your interest in contributing! We are grateful for your work.

Workflow
--------

1. If the work is nontrivial, please make sure there is an  [issue](https://github.com/alleninstitute/transcriptomic_clustering/issues) tracking it. This lets people know that you are interested in addressing this work and gives everyone a chance to weigh in on the appropriate solution.
2. Fork this repository. There should be a current release candidate branch with a name like `rc/X.Y.Z` (there will be a pull request open for the most recent release candidate). You should do your work in a feature branch based on the current release candidate.
3. When you are ready, open a pull request into the release candidate. Before doing so, please make sure that:
    - Any new code has new tests. The success of these tests implies that this code is working as intended.
    - All of the tests pass. Follow the environment and test commands below.
    - You have written documentation for all new code. This means docstrings at the very least. If you are implementing a substantial new feature, consider adding a long-form example notebook demonstrating its use.
    - Your code follows the [Google Python style guide](https://google.github.io/styleguide/pyguide.html).
4. Address any review comments. Once enough reviewers have approved your pull request, a maintainer will merge it.


Testing and supported environments
----------------------------------

CI runs the complete default suite on Linux with Python **3.8 and 3.11**.
Use a separate environment for each interpreter. Python 3.8 remains supported
for existing analyses despite its upstream end of life; prefer 3.11 for new work.
Other Python versions, operating systems and notebooks are not part of this
test matrix.

Install Git LFS and native C/C++ build tools before installing the package.
On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs build-essential cmake pkg-config libxml2-dev
git lfs install
git lfs pull --include="tests/data/**"
git -c lfs.fetchexclude="docs/notebooks/data/**" lfs fsck

python3.11 -m venv .venv  # repeat with python3.8 in a different environment
source .venv/bin/activate
python -m pip install "pip<25.1" "setuptools<76" wheel
python -m pip install -e . -r test_requirements.txt
python -m pip check
python scripts/check_native_graph.py
```

The bootstrap bounds retain Python 3.8 and legacy native build support. CMake
must be on `PATH`; alternatively install `"cmake>=3.18,<4"` in the **activated**
environment. Python 3.11 builds igraph/leidenalg from source and may take several
minutes. Runtime requirements constrain the legacy AnnData/Scanpy numerical
APIs and explicitly pair `igraph==0.9.11`, `python-igraph==0.9.11` and
`leidenalg==0.8.4`. Mixing leidenalg 0.8.4 with igraph 0.11 can abort the
interpreter. Dependencies without known compatibility boundaries are not locked;
record `python -m pip freeze` with results when reproducing an analysis.

Run from the repository root with numerical thread caps (otherwise numerical
libraries can oversubscribe the machine):

```bash
mkdir -p .test-work test-reports
export TMPDIR="$PWD/.test-work"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python -m pytest --collect-only -q
python -m pytest --basetemp=.test-work/pytest \
  --junitxml=test-reports/junit.xml \
  --cov=transcriptomic_clustering --cov-config=.coveragerc \
  --cov-report=term --cov-report=xml:test-reports/coverage.xml
python -m pip freeze > test-reports/environment.txt
```

The full suite includes real chi-square/empirical-Bayes merging, graph
clustering, backed HDF5 and FBM conversion; no slow-test exclusion is needed.
Allow a few minutes and several GB of free disk space. The large fixture payloads
must be downloaded, not replaced with synthetic data or LFS pointer text.
Pytest checks for unresolved pointers before collection and explains how to
download them. Tests seed legacy RNGs and restore their state; new fixtures
should prefer a local `numpy.random.default_rng(seed)`. Close backed
`AnnData.file` handles and keep generated files in pytest-managed directories.

For focused development, use `python -m pytest tests/test_iter_writer.py` (or
the relevant test file); rerun the full suite before submitting. GitHub Actions
uploads JUnit, coverage and resolved environment artifacts for both versions.
The active CircleCI Linux job uses the same requirements and full test command.

Allen Institute Contribution Agreement
--------------------------------------

This document describes the terms under which you may make “Contributions” — 
which may include without limitation, software additions, revisions, bug fixes, configuration changes,
documentation, or any other materials — to any of the projects owned or managed by the Allen Institute.
If you have questions about these terms, please contact us at terms@alleninstitute.org.  

You certify that:

•	Your Contributions are either:

1.	Created in whole or in part by you and you have the right to submit them under the designated license 
(described below); or
2.	Based upon previous work that, to the best of your knowledge, is covered under an appropriate 
open source license and you have the right under that license to submit that work with modifications,
whether created in whole or in part by you, under the designated license; or

3.	Provided directly to you by some other person who certified (1) or (2) and you have not modified them.

•	You are granting your Contributions to the Allen Institute under the terms of the [2-Clause BSD license](https://opensource.org/licenses/BSD-2-Clause)
(the “designated license”).

•	You understand and agree that the Allen Institute projects and your Contributions are public and that 
a record of the Contributions (including all metadata and personal information you submit with them) is 
maintained indefinitely and may be redistributed consistent with the Allen Institute’s mission and the 
2-Clause BSD license.  
