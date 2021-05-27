import pytest

import numpy as np
import pandas as pd
import scanpy as sc

import transcriptomic_clustering as tc

@pytest.fixture
def pair():
    """
        pair of clusters to be compared
    """
    pair = ("1", "2")

    return pair

@pytest.fixture
def cl_present():
    """
        cluster present
    """
    cl_present_1 = np.array([0.05882353,1.00000000,0.76470588,0.17647059,0.17647059,0.94117647,0.35294118,0.00000000,
                        0.00000000,0.88235294,1.00000000,0.00000000,0.64705882,0.94117647,0.70588235,0.58823529])


    cl_present_2 = np.array([0.04761905,0.95238095,0.28571429,0.19047619,0.14285714,0.85714286,0.47619048,0.00000000,
                        0.00000000,0.76190476,0.90476190,0.00000000,0.47619048,0.66666667,0.57142857,0.66666667])

    cl_present = {"1": cl_present_1, "2": cl_present_2}
    
    return cl_present

@pytest.fixture
def cl_means():
    """
        cluster means
    """
    cl_means_1 = np.array([0.10808219,6.14949347,3.66000008,0.71396599,1.15847955,6.07548490,1.64339128,0.00000000,
                     0.00000000,5.31713514,6.71461990,0.00000000,2.47119911,4.47269010,3.49215706,2.66427980])

    cl_means_2 = np.array([0.211184303,6.649927198,1.998193789,0.874202979,0.390651884,5.805244809,2.295554058,
                     0.000000000,0.000000000,4.456795629,6.261517091,0.000000000,3.066860157,3.138447669,
                     3.469266762,3.830073194])

    cl_means = {"1": cl_means_1, "2": cl_means_2}
    
    return cl_means

@pytest.fixture
def cl_size():
    """
        cluster size
    """
    cl_size = {"1":17, "2":21}

    return cl_size

@pytest.fixture
def genes():
    """
        genes
    """
    genes = ['0610005C13Rik','0610007C21Rik','0610007L01Rik','0610007N19Rik','0610007P08Rik','0610007P14Rik',
         '0610007P22Rik','0610008F07Rik','0610009B14Rik','0610009B22Rik','0610009D07Rik','0610009L18Rik',
         '0610009O20Rik','0610010B08Rik','0610010F05Rik','0610010K14Rik']
    
    return genes

