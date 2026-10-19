"""java EndToEndTest. Where the java test writes assertEquals(x, 0,35995f)
it calls the (expected, actual, delta) overload and checks nothing, those
values are tested here with the real outcome instead."""

import numpy as np
import pytest

from conftest import f32, load, round_to_five, search_parameters
from subdisc import condition_list_builder as clb
from subdisc import process
from subdisc.condition import Condition
from subdisc.condition_base import ConditionBase
from subdisc.operators import Operator
from subdisc.qm import QM
from subdisc.quality_measure import QualityMeasure
from subdisc.target_concept import TargetConcept
from subdisc.target_type import TargetType
from subdisc.validation import Validation


def nominal_tc(table):
    tc = TargetConcept()
    tc.set_target_type(TargetType.SINGLE_NOMINAL)
    tc.set_primary_target(table.get_columns()[14])
    tc.set_target_value("gr50K")
    return tc


def numeric_tc(table):
    tc = TargetConcept()
    tc.set_target_type(TargetType.SINGLE_NUMERIC)
    tc.set_primary_target(table.get_columns()[0])   # age
    return tc


def check(s, text, coverage, measure=None, secondary=None, tertiary=None):
    assert str(s) == text
    assert s.get_coverage() == coverage
    if measure is not None:
        assert round_to_five(s.get_measure_value()) == f32(measure)
    if secondary is not None:
        assert round_to_five(s.get_secondary_statistic()) == f32(secondary)
    if tertiary is not None:
        assert round_to_five(s.get_tertiary_statistic()) == f32(tertiary)


def test_adult_1_single_nominal():
    t = load("adult.txt")
    assert (t.get_nr_rows(), t.get_nr_columns()) == (1000, 15)
    sd = process.run_subgroup_discovery(t, 0, None, search_parameters(nominal_tc(t), QM.CORTANA_QUALITY, 0.1), False, 1, None)
    assert t.get_columns()[14].count_values("gr50K", None) == 232
    assert sd.get_number_of_subgroups() == 10
    r = list(sd.get_result())
    check(r[0], "marital-status = 'Married-civ-spouse'", 443, 0.51760, 0.44018, 195.0)
    check(r[1], "relationship = 'Husband'", 376, 0.4533, 0.44681, 168.0)
    check(r[2], "education-num >= 11.0", 327, 0.35996, 0.42813, 140.0)
    check(r[3], "age >= 33.0", 616)
    check(r[4], "hours-per-week >= 43.0", 268)
    check(r[5], "occupation = 'Exec-managerial'", 124)
    check(r[6], "sex = 'Male'", 671)
    check(r[7], "education = 'Bachelors'", 166)
    check(r[8], "capital-gain >= 4386.0", 50)
    check(r[9], "occupation = 'Prof-specialty'", 124, 0.12478, 0.41129, 51.0)


def test_adult_2_z_score():
    t = load("adult.txt")
    sd = process.run_subgroup_discovery(t, 0, None, search_parameters(numeric_tc(t), QM.Z_SCORE, 1.0), False, 1, None)
    assert f32(t.get_columns()[0].get_average(None)) == f32(38.051)
    assert sd.get_number_of_subgroups() == 31
    r = list(sd.get_result())
    check(r[0], "marital-status = 'Widowed'", 33, 8.10605, 56.87879, 13.05220)
    check(r[1], "relationship = 'Husband'", 376, 7.90342)


def test_adult_3_explained_variance():
    t = load("adult.txt")
    sd = process.run_subgroup_discovery(t, 0, None, search_parameters(numeric_tc(t), QM.EXPLAINED_VARIANCE, 0.0), False, 1, None)
    assert sd.get_number_of_subgroups() == 90
    r = list(sd.get_result())
    check(r[0], "marital-status = 'Never-married'", 344, 0.25513, 28.74419, 10.95279)
    check(r[1], "relationship = 'Own-child'", 151, 0.19836, 23.96026, 7.10763)
    check(r[2], "relationship = 'Husband'", 376)
    check(r[89], "marital-status = 'Married-spouse-absent'", 15)


def test_adult_4_swap_randomisation_minimum():
    t = load("adult.txt")
    tc = nominal_tc(t)
    sp = search_parameters(tc, QM.CORTANA_QUALITY, 0.0)
    positives = tc.get_primary_target().count_values("gr50K", None)
    qm = QualityMeasure(sp.get_quality_measure(), t.get_nr_rows(), positives)
    sp.set_quality_measure_minimum(Validation(sp, t, None, qm).get_sign_with_swap_rand(100))
    sd = process.run_subgroup_discovery(t, 0, None, sp, False, 1, None)
    assert sd.get_number_of_subgroups() == 10
    r = list(sd.get_result())
    check(r[0], "marital-status = 'Married-civ-spouse'", 443, 0.51760, 0.44018, 195.0)
    check(r[1], "relationship = 'Husband'", 376, 0.4533, 0.44681, 168.0)


def _male_selection(t):
    cb = ConditionBase(t.get_columns()[9], Operator.EQUALS)
    sel = t.evaluate(clb.create_list(Condition(cb, "Male")))
    assert int(np.count_nonzero(sel)) == 671
    return sel


@pytest.mark.parametrize("kind,qm,minimum", [
    ("nominal", QM.CORTANA_QUALITY, 0.1),
    ("numeric", QM.Z_SCORE, 1.0),
    ("numeric", QM.EXPLAINED_VARIANCE, 0.0),
])
def test_adult_5_to_7_selection_equals_subset_table(kind, qm, minimum):
    large = load("adult.txt")
    small = load("adult_male.txt")
    sel = _male_selection(large)
    make = nominal_tc if kind == "nominal" else numeric_tc
    a = process.run_subgroup_discovery(large, 0, sel, search_parameters(make(large), qm, minimum), False, 1, None)
    b = process.run_subgroup_discovery(small, 0, None, search_parameters(make(small), qm, minimum), False, 1, None)
    assert a.get_number_of_subgroups() == b.get_number_of_subgroups()
    for x, y in zip(a.get_result(), b.get_result()):
        assert str(x) == str(y)
        assert x.get_coverage() == y.get_coverage()
        assert round_to_five(x.get_measure_value()) == round_to_five(y.get_measure_value())
        assert round_to_five(x.get_secondary_statistic()) == round_to_five(y.get_secondary_statistic())
        assert x.get_tertiary_statistic() == y.get_tertiary_statistic()


def test_adult_8_selection_runs():
    t = load("adult.txt")
    sel = _male_selection(t)
    process.run_subgroup_discovery(t, 0, sel, search_parameters(nominal_tc(t), QM.CORTANA_QUALITY, 0.1), False, 1, None)


def test_adult_9_numeric_selection():
    t = load("adult.txt")
    col = t.get_columns()[4]   # education-num
    cb = ConditionBase(col, Operator.GREATER_THAN_OR_EQUAL)
    col.build_sorted(np.zeros(col.size(), dtype=bool))   # empty target, irrelevant here
    i = col.get_sorted_index(10.0)
    sel = t.evaluate(clb.create_list(Condition(cb, 10.0, i)))
    assert int(np.count_nonzero(sel)) == 552
    process.run_subgroup_discovery(t, 0, sel, search_parameters(nominal_tc(t), QM.CORTANA_QUALITY, 0.1), False, 1, None)
