import numpy as np

from conftest import f32, load
from subdisc import condition_list_builder as clb
from subdisc import process
from subdisc.condition import Condition
from subdisc.condition_base import ConditionBase
from subdisc.operators import Operator
from subdisc.qm import QM
from subdisc.search_parameters import SearchParameters
from subdisc.target_concept import TargetConcept


def test_subset_selection():
    t = load("subset selection.txt")
    assert t.get_nr_rows() == 10
    assert t.get_nr_columns() == 5
    target = t.get_columns()[4]

    cb = ConditionBase(t.get_columns()[1], Operator.EQUALS)   # selector 1
    selection = t.evaluate(clb.create_list(Condition(cb, "A")))
    cb = ConditionBase(t.get_columns()[3], Operator.EQUALS)   # descriptor
    subset = t.evaluate(clb.create_list(Condition(cb, "a")))

    tc = TargetConcept()
    tc.set_target_type("single nominal")
    tc.set_primary_target(target)
    tc.set_target_value("t")
    sp = SearchParameters()
    sp.set_target_concept(tc)
    sp.set_quality_measure(QM.CORTANA_QUALITY)
    sp.set_quality_measure_minimum(0.0)
    sp.set_search_depth(1)
    sp.set_minimum_coverage(2)
    sp.set_maximum_coverage_fraction(1.0)
    sp.set_maximum_subgroups(2)
    sp.set_maximum_time(1000)
    sp.set_search_strategy("beam")
    sp.set_nominal_sets(False)
    sp.set_numeric_operators("<html>&#8804;, &#8805;</html>")
    sp.set_numeric_strategy("best")
    sp.set_search_strategy_width(10)
    sp.set_nr_bins(8)
    sp.set_nr_threads(1)
    sd = process.run_subgroup_discovery(t, 0, selection, sp, False, 1, None)

    assert target.count_values("t", None) == 5
    assert int(np.count_nonzero(selection)) == 5   # the first 5 rows
    assert target.count_values("t", selection) == 3
    assert int(np.count_nonzero(subset)) == 5
    assert sd.get_number_of_subgroups() == 2
    it = iter(sd.get_result())
    for expected in ("id <= 2.0", "selector 2 <= 2.0"):
        s = next(it)
        assert str(s) == expected
        assert s.get_coverage() == 2
        assert f32(s.get_measure_value()) == f32(0.6666666865348816)
        assert s.get_secondary_statistic() == 1.0
        assert s.get_tertiary_statistic() == 2.0
