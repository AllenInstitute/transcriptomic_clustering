import logging
import random
from pathlib import Path

import numpy as np
import pytest


def check_fixture_payloads(data_dir):
    if not data_dir.is_dir():
        raise pytest.UsageError(
            'Test fixtures are missing. Run: git lfs pull --include="tests/data/**"'
        )
    pointers = []
    for path in data_dir.rglob("*"):
        if path.is_file():
            with path.open("rb") as stream:
                if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    pointers.append(str(path.relative_to(data_dir)))
    if pointers:
        raise pytest.UsageError(
            "Git LFS fixture payloads are missing: " + ", ".join(pointers) +
            '. Run: git lfs install && git lfs pull --include="tests/data/**"'
        )


def pytest_sessionstart(session):
    check_fixture_payloads(Path(__file__).parent / "data")


@pytest.fixture(autouse=True)
def reproducible_test_state():
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    logger = logging.getLogger("transcriptomic_clustering")
    level = logger.level
    np.random.seed(123)
    random.seed(123)
    logger.setLevel(logging.WARNING)
    yield
    np.random.set_state(numpy_state)
    random.setstate(python_state)
    logger.setLevel(level)


@pytest.fixture
def separated_expression():
    import anndata as ad
    import pandas as pd
    from transcriptomic_clustering import normalize

    rng = np.random.default_rng(42)
    rates = np.broadcast_to(np.geomspace(0.5, 10, 80), (60, 80)).copy()
    rates[:30, :10] = 100
    rates[:30, 10:20] = 0
    rates[30:, :10] = 0
    rates[30:, 10:20] = 100
    counts = rng.poisson(rates).astype(np.float64)
    data = ad.AnnData(
        counts, dtype=counts.dtype,
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(60)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(80)]),
    )
    return normalize(data)
