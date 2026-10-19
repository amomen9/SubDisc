"""Shared bits for the tests. The data files are the ones of the java tests
(Java/src/test/resources), so they are not copied here."""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

RESOURCES = os.path.normpath(os.path.join(HERE, "..", "..", "Java", "src", "test", "resources"))


def resource(name):
    p = os.path.join(RESOURCES, name)
    if not os.path.exists(p):
        pytest.skip("test data not found: " + p)
    return p


def load(name):
    from subdisc.data_loader_txt import DataLoaderTXT
    return DataLoaderTXT(resource(name)).get_table()


def round_to_five(x):
    """(float) Math.round(f*100000)/100000, like the java tests"""
    from subdisc.java_compat import java_round
    return np.float32(np.float32(java_round(float(x) * 100000)) / np.float32(100000))


def f32(x):
    return np.float32(x)


def search_parameters(tc, qm, minimum):
    """the settings every java end-to-end test uses"""
    from subdisc.numeric_operator_setting import NumericOperatorSetting
    from subdisc.numeric_strategy import NumericStrategy
    from subdisc.search_parameters import SearchParameters
    from subdisc.search_strategy import SearchStrategy
    sp = SearchParameters()
    sp.set_target_concept(tc)
    sp.set_quality_measure(qm)
    sp.set_quality_measure_minimum(minimum)
    sp.set_search_depth(1)
    sp.set_minimum_coverage(2)
    sp.set_maximum_coverage_fraction(1.0)
    sp.set_maximum_subgroups(1000)
    sp.set_maximum_time(1000)   # 1000 seconds
    sp.set_search_strategy(SearchStrategy.BEAM)
    sp.set_nominal_sets(False)
    sp.set_numeric_operators(NumericOperatorSetting.NORMAL)
    sp.set_numeric_strategy(NumericStrategy.NUMERIC_BEST)
    sp.set_search_strategy_width(10)
    sp.set_nr_bins(8)
    sp.set_nr_threads(1)
    return sp
