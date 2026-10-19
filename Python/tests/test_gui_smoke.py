"""Opens the main window offscreen, mines, and opens the result windows.
Not in java (there were no gui tests), skipped without PySide6."""

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import resource   # noqa: E402


@pytest.fixture(scope="module")
def app():
    from subdisc.gui.app import create_app
    return create_app()


def _visible(name):
    from PySide6.QtWidgets import QApplication
    return [w for w in QApplication.topLevelWidgets() if w.isVisible() and type(w).__name__ == name]


def test_mining_window_flow(app):
    from subdisc import process
    from subdisc.file_handler import Action, FileHandler
    from subdisc.gui.mining_window import MiningWindow
    t = FileHandler(Action.OPEN_FILE, resource("adult.txt")).get_table()
    w = MiningWindow(t)
    w.show()
    assert w._c_target_type.currentText() == "single nominal"
    assert w._c_target_attribute.currentText() == "target"
    assert w._t_coverage_min.text() == "100"
    w._setup_search_parameters()
    sd = process.run_subgroup_discovery(t, 0, None, w._search_parameters, True, 1, w)
    app.processEvents()
    results = _visible("ResultWindow")
    assert results and results[-1]._model.rowCount() == len(sd.get_result())
    r = results[-1]
    r._roc_clicked()
    r._browse_clicked()
    app.processEvents()
    assert _visible("ROCCurveWindow") and _visible("BrowseWindow")
    for x in _visible("ROCCurveWindow") + _visible("BrowseWindow") + results:
        x.close()
    w.hide()


def test_target_type_cascade(app):
    from subdisc.file_handler import Action, FileHandler
    from subdisc.gui.mining_window import MiningWindow
    t = FileHandler(Action.OPEN_FILE, resource("adult.txt")).get_table()
    w = MiningWindow(t)
    w._select(w._c_target_type, "double regression")
    w._target_type_changed()
    assert w._c_misc_field.isVisible() or not w.isVisible()
    assert w._l_target_info.text().strip() == "regression"
    w._select(w._c_target_type, "single numeric")
    w._target_type_changed()
    assert w._l_target_info.text().strip() == "average"
    assert w._c_numeric_strategy.findText("intervals") < 0
    w.hide()
