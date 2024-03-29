import pytest
import os
import numpy as np
import pandas as pd

import transcriptomic_clustering as tc
from transcriptomic_clustering import diff_expression as de

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

@pytest.fixture
def pair():
    """
        pair of clusters to be compared
    """
    pair = ("1", "2")

    return pair

@pytest.fixture
def genes():
    """
        genes
    """
    genes = ['0610005C13Rik','0610007C21Rik','0610007L01Rik','0610007N19Rik','0610007P08Rik','0610007P14Rik',
         '0610007P22Rik','0610008F07Rik','0610009B14Rik','0610009B22Rik','0610009D07Rik','0610009L18Rik',
         '0610009O20Rik','0610010B08Rik','0610010F05Rik','0610010K14Rik']

    return genes

@pytest.fixture
def cl_present(genes):
    """
        cluster present (clusters x genes)
    """
    cl_present_1 = np.array([0.05882353,1.00000000,0.76470588,0.17647059,0.17647059,0.94117647,0.35294118,0.00000000,
                        0.00000000,0.88235294,1.00000000,0.00000000,0.64705882,0.94117647,0.70588235,0.58823529])


    cl_present_2 = np.array([0.04761905,0.95238095,0.28571429,0.19047619,0.14285714,0.85714286,0.47619048,0.00000000,
                        0.00000000,0.76190476,0.90476190,0.00000000,0.47619048,0.66666667,0.57142857,0.66666667])

    cl_present = pd.DataFrame(np.array([cl_present_1, cl_present_2]).T, columns=["1","2"], index = genes)
    
    return cl_present.T

@pytest.fixture
def cl_means(genes):
    """
        cluster means (clusters x genes)
    """
    cl_means_1 = np.array([0.10808219,6.14949347,3.66000008,0.71396599,1.15847955,6.07548490,1.64339128,0.00000000,
                     0.00000000,5.31713514,6.71461990,0.00000000,2.47119911,4.47269010,3.49215706,2.66427980])

    cl_means_2 = np.array([0.211184303,6.649927198,1.998193789,0.874202979,0.390651884,5.805244809,2.295554058,
                     0.000000000,0.000000000,4.456795629,6.261517091,0.000000000,3.066860157,3.138447669,
                     3.469266762,3.830073194])

    cl_means = pd.DataFrame(np.array([cl_means_1, cl_means_2]).T, columns=["1","2"], index = genes)
    
    return cl_means.T

@pytest.fixture
def cl_size():
    """
        cluster size
    """
    cl_size = {"1":17, "2":21}

    return cl_size

@pytest.fixture
def expected_chisq_pair_statistics(genes):
    """
        expected pairwise chisq statistics result
    """
    p_adj = np.array([1., 1., 0.144849, 1., 1.,1., 1., 1., 1., 1.,
                1., 1., 1., 1, 1.,1.])
    p_value = np.array([1., 1., 0.009053, 1., 1., 0.758282,0.663754, 1., 1., 0.595042, 0.564113, 1.,
                    0.468314, 0.096169, 0.605741, 0.872733])
    lfc = np.array([-0.10310211, -0.50043373,  1.66180629, -0.16023699,  0.76782767,
                    0.27024009, -0.65216278,  0.,  0.,  0.86033951, 0.45310281,  0., 
                    -0.59566105,  1.33424243,  0.0228903, -1.16579339])
    meanA = np.array([0.10808219, 6.14949347, 3.66000008, 0.71396599, 1.15847955,
                    6.0754849 , 1.64339128, 0.        , 0.        , 5.31713514,
                    6.7146199 , 0.        , 2.47119911, 4.4726901 , 3.49215706,
                    2.6642798 ])
    meanB = np.array([0.2111843 , 6.6499272 , 1.99819379, 0.87420298, 0.39065188,
                    5.80524481, 2.29555406, 0., 0., 4.45679563,6.26151709, 0., 3.06686016, 
                    3.13844767, 3.46926676,3.83007319])
    q1 = np.array([0.05882353, 1., 0.76470588, 0.17647059, 0.17647059,
                0.94117647, 0.35294118, 0., 0., 0.88235294, 1., 0., 0.64705882, 
                0.94117647, 0.70588235, 0.58823529])
    q2 = np.array([0.04761905, 0.95238095, 0.28571429, 0.19047619, 0.14285714,
                0.85714286, 0.47619048, 0., 0., 0.76190476, 0.9047619 , 0., 0.47619048, 
                0.66666667, 0.57142857, 0.66666667])

    qdiff = np.array([0.19047616, 0.04761905, 0.62637362, 0.0735294,  0.19047621, 0.08928571,
                0.25882353, 0., 0., 0.13650794, 0.0952381, 0.,  0.26406925, 0.29166666,
                0.19047619, 0.11764707])

    expected_chisq_result = pd.DataFrame(
        {
            "gene": genes,
            "p_adj": p_adj,
            "p_value": p_value,
            "lfc": lfc,
            "meanA": meanA,
            "meanB": meanB,
            "q1": q1,
            "q2": q2,
            "qdiff": qdiff
        })
    expected_chisq_result.set_index("gene")

    return expected_chisq_result


@pytest.fixture
def de_stats():
    de_stats_csv = os.path.join(DATA_DIR, "de_stats.csv")

    return pd.read_csv(de_stats_csv, index_col='gene')


@pytest.fixture
def thresholds():
    thresholds = {
        'q1_thresh': 0.5,
        'q2_thresh': 0.7,
        'cluster_size_thresh': 4,
        'qdiff_thresh': 0.7,
        'padj_thresh': 0.01,
        'lfc_thresh': 1.0
    }
    return thresholds

def test_vec_chisq_test(pair, cl_present,cl_size, expected_chisq_pair_statistics):
    """
        test vec_chisq_test func
    """
    pair = pair
    cl_present = cl_present
    cl_size = cl_size

    expected_chisq_result = expected_chisq_pair_statistics
    expected_p_vals = expected_chisq_result['p_value'].to_numpy()

    p_vals = de.vec_chisq_test(pair,
                            cl_present,
                            cl_size)

    np.testing.assert_allclose(p_vals,
                            expected_p_vals,
                            rtol=1e-06,
                            atol=1e-06,
                            )


def test_de_pair_chisq(pair, cl_present, cl_means, cl_size, expected_chisq_pair_statistics):
    """
        test de_pair_chisq func
    """

    pair = pair
    cl_present = cl_present
    cl_means = cl_means
    cl_size = cl_size


    expected_de_stats = expected_chisq_pair_statistics
    obtained_de_stats = de.de_pair_chisq(pair, cl_present, cl_means, cl_size)

    pd.testing.assert_frame_equal(expected_de_stats, obtained_de_stats)


def test_filter_up_regulated_gene_stats(de_stats,thresholds):

    expected_genes = ['Elovl2', 'Iqsec1', 'Prdm16']

    obtained_genes = de.filter_gene_stats(
        de_stats=de_stats,
        gene_type='up-regulated',
        cl1_size=10,
        cl2_size=5,
        **thresholds
    ).index.values

    assert set(expected_genes) == set(obtained_genes)


def test_filter_down_regulated_gene_stats(de_stats,thresholds):

    expected_genes = ['Fam5c', '3632451O06Rik', 'Tspan2', 'Bcas1', 'Qpct']

    obtained_genes = de.filter_gene_stats(
        de_stats=de_stats,
        gene_type='down-regulated',
        cl1_size=10,
        cl2_size=5,
        **thresholds
    ).index.values

    assert set(expected_genes) == set(obtained_genes)


def test_get_qdiff():

    expected = np.array([(0.6 - 0.4)/0.6])
    obtained = de.get_qdiff(np.array([0.4]), np.array([0.6]))
    assert np.isclose(expected, obtained)


def test_calc_de_score():

    padj = np.array([0.5, 0.5])
    expected_score = -2*np.log10(0.5)
    obtained_score = de.calc_de_score(padj)
    assert np.isclose(expected_score,obtained_score)


def test_calc_de_score_for_pair(de_stats,thresholds):

    de_stats_up_genes = de.filter_gene_stats(
        de_stats=de_stats,
        gene_type='up-regulated',
        cl1_size=10,
        cl2_size=5,
        **thresholds
    )
    up_score = de.calc_de_score(de_stats_up_genes['p_adj'])

    de_stats_down_genes = de.filter_gene_stats(
        de_stats=de_stats,
        gene_type='down-regulated',
        cl1_size=10,
        cl2_size=5,
        **thresholds
    )
    down_score = de.calc_de_score(de_stats_down_genes['p_adj'])

    expected_score = 16.936848586
    obtained_score = up_score + down_score

    assert np.isclose(expected_score, obtained_score)


def test_de_pairs_chisq():

    de_thresholds = {
        'q1_thresh': 0.3,
        'q2_thresh': None,
        'cluster_size_thresh': 1,
        'qdiff_thresh': 0.1,
        'padj_thresh': 0.5,
        'lfc_thresh': .4,
    }
    pairs = [(11, 4), (1, 2)]
    cl_size = {4: 100, 11: 200, 1: 17, 2: 21}

    genes = ['gene_a', 'gene_b', 'gene_c', 'gene_d', 'gene_e']

    cluster_means = pd.DataFrame(
        np.array([[3.8, 1.5, 4., 5., 3.],
                [3., 2., 3., 1., 3.],
                [1., 4., 2., 3., 2.],
                [3., 1., 1., 2., 1.]]),
        index=[11, 4, 1, 2],
        columns=genes)

    present_cluster_means = pd.DataFrame(
        np.array([[.85, .67, .51, .14, .23],
                [.1, .6, .4, .3, .2],
                [.2, .16, .5, .11, .71],
                [.3, .44, .2, .32, .83]]),
        index=[11, 4, 1, 2],
        columns=genes)

    expected_score = 34.29963735220708

    expected_up_genes_index = [0, 2]

    de_pairs = de.de_pairs_chisq(pairs,
                                cluster_means,
                                present_cluster_means,
                                cl_size,
                                de_thresholds)

    np.testing.assert_almost_equal(de_pairs.loc[(11,4)]['score'], expected_score, decimal=10)
    assert set(de_pairs.loc[(11,4)]['up_genes']) == set(expected_up_genes_index)