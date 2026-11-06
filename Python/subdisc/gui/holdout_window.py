"""Holdout Test: the subgroups of a result window applied to rows they were
not mined on, train and test numbers next to each other. Not in java."""

import numpy as np

from .. import holdout
from .. import result_table_model as rtm
from . import common
from .feature_common import TableResultWindow, html_table

HEADERS = ("Nr.", "Conditions", "Train Coverage", "Test Coverage", "Train Quality", "Test Quality", "Drop")


def open_holdout_window(parent, subgroups, search_parameters, train_rows, choice):
    """choice from TestDataChooser.choose()"""
    kind, eval_table, eval_rows = choice
    tt = search_parameters.get_target_type()
    stat = rtm.get_column_name(4, tt)
    common.set_busy(True)
    try:
        rows = holdout.score_on(subgroups, search_parameters, eval_table, eval_rows)
    finally:
        common.set_busy(False)
    n_test = eval_table.get_nr_rows() if eval_rows is None else int(np.count_nonzero(eval_rows))
    table_rows = [(r.subgroup.get_id(), str(r.subgroup.get_conditions()), r.train_coverage, r.test_coverage,
                   r.train_quality, r.test_quality, r.drop(), r.train_secondary, r.test_secondary) for r in rows]
    s = holdout.summary(rows, train_rows, n_test, search_parameters.get_quality_measure_minimum())
    where = ("rows of this table that were not in the mining selection" if kind == "rest"
             else "the file '%s'" % eval_table.get_source())
    html = ("<b>Tested on %s.</b> Quality measure: %s. A large positive drop means the subgroup was partly luck. "
            "The rank correlation says whether the order of the subgroups holds on the test data.<br>%s"
            % (where, search_parameters.get_quality_measure().GUI_TEXT, html_table(s)))
    w = TableResultWindow("Holdout Test - %d subgroups" % len(rows), HEADERS + ("Train " + stat, "Test " + stat),
                          table_rows, html, "subgroups", preset_kind="holdout")
    return w
