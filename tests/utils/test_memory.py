import pytest
from unittest import mock
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import scanpy as sc

import transcriptomic_clustering as tc


@pytest.fixture
def test_memory(monkeypatch):
    # limit of 100GB, 70GB available, 50GB used (120 GB total)
    def mock_virtual_memory():
        mock_vm_result = mock.MagicMock()
        mock_vm_result.available = (70 * 1024 ** 3)
        return mock_vm_result
    monkeypatch.setattr('psutil.virtual_memory', mock_virtual_memory)

    def mock_process():
        mock_memory_result = mock.MagicMock()
        mock_memory_result.rss = 50 * 1024 ** 3
        mock_process_result = mock.MagicMock()
        mock_process_result.memory_info = mock.MagicMock(return_value=mock_memory_result)
        return mock_process_result
    monkeypatch.setattr('psutil.Process', mock_process) 
    
    test_memory = tc.utils.memory.Memory()
    test_memory.set_memory_limit(GB=100)
    test_memory.allow_chunking = True
    return test_memory

@pytest.fixture
def test_adata(monkeypatch):
    monkeypatch.setattr('scanpy.AnnData.isbacked', True)
    return sc.AnnData(np.zeros((40,1)))


def test_set_memory_GB(test_memory):
    test_memory.set_memory_limit(GB=100)
    assert test_memory.memory_limit_GB == 100

def test_set_memory_percent(test_memory):
    # 100 limit, 50 used, 70 available on system.
    # updates limit to 50% of 70 = 35 GB
    test_memory.set_memory_limit(percent_current_available=50)
    assert test_memory.memory_limit_GB == 35

def test_get_available_memory_no_limit(test_memory):
    # 70 available, no limit
    test_memory.remove_memory_limit()
    assert test_memory.get_available_memory_GB() == 70

def test_get_available_memory_with_limit(test_memory):
    # 75 on machine, but only 50 left till limit
    assert test_memory.get_available_memory_GB() == 50

def test_estimate_n_chunks(test_memory):
    assert test_memory.estimate_n_chunks(500) == 10

def test_estimate_n_chunks_w_output(test_memory):
    assert test_memory.estimate_n_chunks(500, 25) == 20

def test_estimate_n_chunks_percent(test_memory):
    assert test_memory.estimate_n_chunks(120, percent_allowed=80) == 3

def test_estimate_n_chunks_not_enough(test_memory):
    with pytest.raises(MemoryError):
        test_memory.estimate_n_chunks(0, 55)        

def test_chunking_not_allowed(test_memory):
    test_memory.allow_chunking = False
    with pytest.raises(MemoryError):
        test_memory.estimate_n_chunks(500)

def test_too_many_chunks(test_memory):
    with pytest.raises(MemoryError):
        # 50 GB available, 49 for output, 1 GB left for processing
        # 5001 GB requires 5001 chunks > max_chunks = 5000
        test_memory.estimate_n_chunks(5001,49)

def test_estimate_chunk_size(test_memory, test_adata):
    assert test_memory.estimate_chunk_size(test_adata, 500, 25) == 2

def test_import():
    import transcriptomic_clustering as tc
    tc.memory.set_memory_limit(GB=50)
    import transcriptomic_clustering as tc
    assert tc.memory.memory_limit_GB == 50