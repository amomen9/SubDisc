"""Stability: mines again on random subsamples and counts how often every
subgroup of a result window comes back (see stability.py). Not in java."""

from .. import stability
from . import common
from .feature_common import SettingsDialog, TableResultWindow, html_table

HEADERS = ("Nr.", "Conditions", "Coverage", "Quality", "Same Description", "Same Rows", "Mean Jaccard")


def start_stability(parent, table, selection, subgroups, search_parameters):
    d = SettingsDialog(parent, "Stability",
                       "Runs the same search again on random subsamples of the rows and checks, for every one of the "
                       "%d subgroups shown, how often it is found again. 'Same rows' allows the description to "
                       "shift a little (age >= 33 against age >= 34) as long as the rows stay nearly the same." % len(subgroups))
    d.add_int("runs", "Number of runs", 10, 2, 500)
    d.add_float("fraction", "Rows per run (fraction)", 0.8, 0.1, 0.99, 2, 0.05)
    d.add_float("tau", "'Same rows' when Jaccard >=", 0.8, 0.5, 1.0, 2, 0.05)
    d.add_int("seed", "Random seed", 1, 0, 2 ** 31 - 1)
    v = d.values()
    if v is None:
        return
    task_box = {}

    def progress(text):
        t = task_box.get("task")
        if t is not None:
            common.bridge().call(t.status.setText, text)

    canceller = stability.Canceller(progress)

    def work():
        return stability.run(table, selection, search_parameters, subgroups, v["runs"], v["fraction"], v["seed"],
                             v["tau"], canceller)

    def done(res):
        if res is None:
            return
        rows, stats = res
        table_rows = [(r.subgroup.get_id(), str(r.subgroup.get_conditions()), r.subgroup.get_coverage(),
                       float(r.subgroup.get_measure_value()), r.exact, r.cover, r.mean_jaccard) for r in rows]
        stats["rows per run"] = "%d (%.0f%%)" % (stats["rows per run"], 100 * v["fraction"])
        html = ("<b>Subsampling stability.</b> 'Same Description' and 'Same Rows' are the share of runs (0 to 1) in "
                "which the subgroup came back. Low values: the subgroup leans on a few rows.<br>" + html_table(stats))
        if canceller.cancelled:
            html = "<b>Stopped early</b>, only the finished runs count.<br>" + html
        TableResultWindow("Stability - %d runs" % stats["runs done"], HEADERS, table_rows, html, "subgroups",
                          preset_kind="stability")

    task = common.BackgroundTask(parent, "Stability...", "Mining %d subsamples. Stop keeps the runs done so far." % v["runs"],
                                 work, done, lambda: setattr(canceller, "cancelled", True))
    task_box["task"] = task
    task.start()
