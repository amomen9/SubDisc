"""The windows of the new features (NEW_FEATURES.md), offscreen. The modal
dialogs are replaced by their answers so nothing waits for a click."""

import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import resource   # noqa: E402

DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))


@pytest.fixture(scope="module")
def app():
    from subdisc.gui.app import create_app
    return create_app()


def _visible(name):
    from PySide6.QtWidgets import QApplication
    return [w for w in QApplication.topLevelWidgets() if w.isVisible() and type(w).__name__ == name]


def _close_all(*names):
    for n in names:
        for w in _visible(n):
            w.close()


def _wait_for(app, cond, seconds=60):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def _survival_table():
    from subdisc.file_handler import Action, FileHandler
    p = os.path.join(DATA, "survival_demo.txt")
    if not os.path.exists(p):
        pytest.skip("no survival_demo.txt")
    return FileHandler(Action.OPEN_FILE, p).get_table()


def test_survival_target_in_the_main_window(app):
    from subdisc import process
    from subdisc.gui.mining_window import MiningWindow
    t = _survival_table()
    w = MiningWindow(t)
    w.show()
    w._select(w._c_target_type, "survival (time-to-event)")
    w._target_type_changed()
    assert w._c_target_attribute.currentText() == "time"
    assert w._c_misc_field.currentText() == "event"
    assert "events" in w._l_target_info_text.text()
    assert w._c_quality_measure.findText("Log-rank test") >= 0
    w._base_model()
    app.processEvents()
    assert _visible("KaplanMeierWindow")
    w._select(w._c_quality_measure, "Log-rank z (worse survival)")
    w._quality_measure_changed()
    w._t_depth.setText("2")
    w._setup_search_parameters()
    sd = process.run_subgroup_discovery(t, 0, None, w._search_parameters, True, 1, w)
    app.processEvents()
    r = _visible("ResultWindow")[-1]
    assert r._model.rowCount() == len(sd.get_result())
    assert r._model.header(4) == "Hazard Ratio" and r._model.header(5) == "Median Time"
    r._view.selectRow(0)
    r._show_model_clicked()
    app.processEvents()
    assert len(_visible("KaplanMeierWindow")) == 2
    # multiple testing has an exact test (log-rank) for this target type
    r._multiple_testing_clicked()
    app.processEvents()
    mtw = _visible("MultipleTestingWindow")[-1]
    assert mtw.model().rowCount() == len(sd.get_result())
    for i in range(mtw._method.count()):
        mtw._method.setCurrentIndex(i)
    _close_all("KaplanMeierWindow", "MultipleTestingWindow", "ResultWindow")
    w.hide()


def test_result_window_new_buttons(app, monkeypatch):
    from subdisc import process
    from subdisc.file_handler import Action, FileHandler
    from subdisc.gui import feature_common
    from subdisc.gui.feature_common import SettingsDialog
    from subdisc.gui.holdout_window import open_holdout_window
    from subdisc.gui.mining_window import MiningWindow
    from subdisc import holdout
    t = FileHandler(Action.OPEN_FILE, resource("adult.txt")).get_table()
    w = MiningWindow(t)
    w.show()
    # Tools > Random Holdout Split
    import subdisc.gui.model_error_dialog as med
    train, test = holdout.random_split(t.get_nr_rows(), 0.3, 4)
    monkeypatch.setattr(med, "ask_holdout_split", lambda parent, table: train)
    w._random_holdout_split()
    assert int(np.count_nonzero(w._selection)) == 700
    assert "700 in selection" in w._l_nr_examples.text()
    w._setup_search_parameters()
    sd = process.run_subgroup_discovery(t, 0, w._selection, w._search_parameters, True, 1, w)
    app.processEvents()
    r = _visible("ResultWindow")[-1]

    # Holdout Test on the rows left out
    choice = ("rest", t, ~w._selection)
    hw = open_holdout_window(r, list(sd.get_result()), r._sp_run, 700, choice)
    assert hw.model().rowCount() == len(sd.get_result())

    # Subgroup List, the dialog answers for itself
    monkeypatch.setattr(SettingsDialog, "values", lambda self: {"gain": "absolute (bits)", "max": 20, "min_rows": 1})
    r._subgroup_list_clicked()
    app.processEvents()
    slw = _visible("SubgroupListWindow")[-1]
    assert slw.model().rowCount() == len(slw._sl.entries) + 1
    monkeypatch.setattr(feature_common.TestDataChooser, "choose", lambda self: choice)
    slw._test()
    assert "bits saved on test" in slw._summary.text()

    # Stability runs on a worker thread
    monkeypatch.setattr(SettingsDialog, "values", lambda self: {"runs": 3, "fraction": 0.8, "tau": 0.8, "seed": 1})
    r._stability_clicked()
    assert _wait_for(app, lambda: any(x.windowTitle().startswith("Stability") for x in _visible("TableResultWindow")))

    # Multiple Testing with the exact test
    r._multiple_testing_clicked()
    app.processEvents()
    assert _visible("MultipleTestingWindow")

    # a filter in the result window limits what the new buttons use
    _close_all("MultipleTestingWindow")
    app.processEvents()
    r._model.set_filter(2, ">=", "300")
    shown = r._model.rowCount()
    assert 0 < shown < len(sd.get_result())
    r._multiple_testing_clicked()
    app.processEvents()
    windows = _visible("MultipleTestingWindow")
    assert len(windows) == 1 and windows[0].model().rowCount() == shown
    _close_all("TableResultWindow", "SubgroupListWindow", "MultipleTestingWindow", "ResultWindow")
    w.hide()


def test_model_error_columns_from_the_tools_menu(app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from subdisc.file_handler import Action, FileHandler
    from subdisc.gui.feature_common import SettingsDialog
    from subdisc.gui.mining_window import MiningWindow
    t = FileHandler(Action.OPEN_FILE, resource("adult.txt")).get_table()
    w = MiningWindow(t)
    w.show()
    monkeypatch.setattr(SettingsDialog, "values", lambda self: {
        "target": "target", "value": "gr50K", "folds": 5, "l2": 1.0, "threshold": 0.5, "seed": 1, "disable": True})
    monkeypatch.setattr(QMessageBox, "exec", lambda self: 0)
    w._add_model_error_columns()
    assert t.get_column("model_error") is not None
    assert w._c_target_attribute.currentText() == "model_error"
    assert w._c_misc_field.currentText() == "1"
    assert not t.get_column("model_score").get_is_enabled()
    assert not t.get_column("target").get_is_enabled()
    w.hide()
