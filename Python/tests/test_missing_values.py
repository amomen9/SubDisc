from conftest import f32, load, round_to_five, search_parameters
from subdisc import process
from subdisc.qm import QM
from subdisc.target_concept import TargetConcept
from subdisc.target_type import TargetType


def test_adult_single_nominal():
    t = load("adult with missing.txt")
    assert t.get_nr_rows() == 1000
    assert t.get_nr_columns() == 15
    target = t.get_columns()[14]
    tc = TargetConcept()
    tc.set_target_type(TargetType.SINGLE_NOMINAL)
    tc.set_primary_target(target)
    tc.set_target_value("gr50K")
    # disable most columns, keep 0 (age) and 5 (marital-status)
    for i in range(1, 15):
        if i != 5:
            t.get_columns()[i].set_is_enabled(False)
    sp = search_parameters(tc, QM.CORTANA_QUALITY, -1.0)
    sd = process.run_subgroup_discovery(t, 0, None, sp, False, 1, None)

    assert target.count_values("gr50K", None) == 232
    assert sd.get_number_of_subgroups() == 7
    it = iter(sd.get_result())
    s = next(it)
    assert str(s) == "marital-status = 'Married-civ-spouse'"
    assert s.get_coverage() == 444
    assert round_to_five(s.get_measure_value()) == f32(0.5163)
    assert round_to_five(s.get_secondary_statistic()) == f32(0.43919)
    assert s.get_tertiary_statistic() == 195.0
    s = next(it)
    assert str(s) == "age >= 37.0"
    assert s.get_coverage() == 498
    assert round_to_five(s.get_measure_value()) == f32(0.33374)
    assert s.get_tertiary_statistic() == 175.0


def test_adult_single_numeric_average():
    t = load("adult with missing.txt")
    assert t.get_nr_rows() == 1000
    assert t.get_nr_columns() == 15
    tc = TargetConcept()
    tc.set_target_type(TargetType.SINGLE_NUMERIC)
    tc.set_primary_target(t.get_columns()[14])
    assert round_to_five(t.get_columns()[0].get_average(None)) == f32(45.70549)
