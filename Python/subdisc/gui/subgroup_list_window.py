"""Subgroup List: an ordered, non overlapping list picked from the subgroups
of a result window with MDL (see subgroup_list.py). Not in java."""

import numpy as np
from PySide6.QtWidgets import QMessageBox

from .. import holdout
from ..subgroup_list import SubgroupList
from . import common
from .feature_common import SettingsDialog, TableResultWindow, TestDataChooser, html_table


def open_subgroup_list(parent, table, selection, subgroups, search_parameters):
    kind = holdout.target_kind(search_parameters)
    if kind is None:
        QMessageBox.information(parent, "Subgroup List", "A subgroup list needs a single nominal or single numeric target.")
        return None
    d = SettingsDialog(parent, "Subgroup List",
                       "Picks, from the %d subgroups shown, an ordered list in which every row belongs to the first "
                       "subgroup that covers it. A subgroup only goes in when it saves more bits on the target than "
                       "it costs to describe (MDL)." % len(subgroups))
    d.add_combo("gain", "Gain", ("absolute (bits)", "normalised (bits per row)"))
    d.add_int("max", "Maximum list length", 20, 1, 1000)
    d.add_int("min_rows", "Minimum new rows per subgroup", max(1, search_parameters.get_minimum_coverage()), 1, 10 ** 7)
    v = d.values()
    if v is None:
        return None
    tc = search_parameters.get_target_concept()
    y = holdout.target_values(table, tc)
    n = table.get_nr_rows()
    rows = np.ones(n, dtype=bool) if selection is None else np.asarray(selection[:n], dtype=bool)
    common.set_busy(True)
    try:
        sl = SubgroupList.from_subgroups(subgroups, kind, y, rows, table, search_parameters.get_nr_bins(),
                                         v["gain"].startswith("normalised"), v["max"], v["min_rows"])
    except ValueError as e:
        common.set_busy(False)
        QMessageBox.warning(parent, "Subgroup List", str(e))
        return None
    finally:
        common.set_busy(False)
    return SubgroupListWindow(table, selection, sl, search_parameters)


class SubgroupListWindow(TableResultWindow):
    def __init__(self, table, selection, sl, search_parameters):
        self._table = table
        self._selection = selection
        self._sl = sl
        self._sp = search_parameters
        stat = "Target Share" if sl.kind == "binary" else "Average"
        headers = ["Position", "Nr.", "Conditions", "Coverage", "New Rows", stat]
        if sl.kind == "numeric":
            headers.append("St. Dev.")
        headers += ["Saved Bits", "Model Bits", "Gain"]
        test = common.build_button("Test on Held Out Data...", "T", self._test)
        super().__init__("Subgroup List - %d subgroups + default rule" % len(sl.entries), headers, self._rows(),
                         self._summary_html(), "subgroups", preset_kind="subgroup_list", extra_buttons=[test])

    def _rows(self):
        out = []
        sl = self._sl
        for k, e in enumerate(sl.entries, 1):
            r = [k, e.subgroup.get_id(), str(e.subgroup.get_conditions()), e.rows, e.new_rows, e.statistic]
            if sl.kind == "numeric":
                r.append(e.spread)
            r += [e.saved_bits, e.model_bits, e.gain]
            out.append(r)
        r = [len(sl.entries) + 1, "", "default rule (everything else)", sl.nr_rows, sl.default_rows, sl.default_statistic]
        if sl.kind == "numeric":
            r.append(sl.default_spread)
        r += [0.0, 0.0, 0.0]
        out.append(r)
        return out

    def _summary_html(self, extra=None):
        sl = self._sl
        info = [("subgroups in the list", len(sl.entries)),
                ("rows", sl.nr_rows),
                ("rows left for the default rule", sl.default_rows),
                ("target with the default rule alone (bits)", sl.default_only_bits),
                ("total gain of the list (bits)", sl.total_gain),
                ("compression", sl.compression())]
        html = ("<b>MDL subgroup list.</b> Read it top to bottom: a row belongs to the first subgroup that covers it. "
                "'New Rows' are the rows a subgroup takes over from the ones below it.<br>" + html_table(info))
        if extra:
            html += "<br>" + extra
        return html

    def _test(self):
        choice = TestDataChooser(self, self._table, self._selection, "the subgroup list").choose()
        if choice is None:
            return
        kind, eval_table, eval_rows = choice
        try:
            sp = holdout.map_search_parameters(self._sp, eval_table)
            y = holdout.target_values(eval_table, sp.get_target_concept())
            common.set_busy(True)
            res = self._sl.score(eval_table, y, eval_rows)
        except Exception as e:
            common.set_busy(False)
            QMessageBox.warning(self, "Test", str(e))
            return
        finally:
            common.set_busy(False)
        where = "held out rows" if kind == "rest" else "'%s'" % eval_table.get_source()
        self.set_summary(self._summary_html("<b>On %s</b> (parameters from the training rows, nothing refitted; "
                                            "'bits saved' above 0 means the list predicts better than the overall rate):<br>%s"
                                            % (where, html_table(res))))
