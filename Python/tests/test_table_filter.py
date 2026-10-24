"""Sorting / filtering of the gui tables (not in java). Offscreen, skipped
without PySide6."""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from conftest import resource   # noqa: E402


@pytest.fixture(scope="module")
def app():
    from subdisc.gui.app import create_app
    return create_app()


@pytest.fixture
def presets_store(tmp_path):
    """presets go to a temporary ini file, not to the real settings"""
    from subdisc.gui import table_filter as tf
    old = tf.PRESET_STORE
    tf.PRESET_STORE = str(tmp_path / "presets.ini")
    yield
    tf.PRESET_STORE = old


def test_match_numbers_text_and_operators():
    from subdisc.gui.table_filter import match
    assert match(5178.0, "5,178", ">=", "5,000")
    assert match(0.5176005, "0.517601", "=", "0.517601")   # as shown
    assert not match(float("nan"), "  -", ">", "0")
    assert match(None, "Married-civ-spouse", "include", "civ")
    assert match(None, "age >= 33.0", "exclude", "sex")
    assert match(None, "Bachelors", "=", "bachelors")       # ignores case
    assert match(None, "b", ">", "a")


def test_two_conditions_and_or():
    from subdisc.gui.table_filter import ColumnFilter
    between = ColumnFilter(">=", "100", "and", "<=", "300")
    assert between.accepts(200, "200") and not between.accepts(400, "400")
    either = ColumnFilter("include", "age", "or", "include", "sex")
    assert either.accepts(None, "sex = 'Male'") and either.accepts(None, "age >= 33.0")
    assert not either.accepts(None, "race = 'White'")
    # an empty second value means one condition
    assert len(ColumnFilter("=", "1", "or", "=", "  ").conds) == 1
    assert ColumnFilter.from_json(between.to_json()) == between
    assert between.describe() == ">= 100 and <= 300"


def test_list_model_filter_or_and_sort(app):
    from PySide6.QtCore import Qt
    from subdisc.gui.table_filter import ListTableModel
    m = ListTableModel(["n", "name"], [[3, "c"], [1, "a"], [2, "b"], [10, "j"]])
    m.set_filter(0, "<", "2", "or", ">", "5")
    assert [m.display(m.source_row(i), 0) for i in range(m.rowCount())] == ["1", "10"]
    m.clear_filters()
    m.sort(0, Qt.DescendingOrder)
    assert [m.raw(m.source_row(i), 0) for i in range(m.rowCount())] == [10, 3, 2, 1]
    m.sort(-1)
    assert [m.raw(m.source_row(i), 0) for i in range(m.rowCount())] == [3, 1, 2, 10]


def test_browse_model_numpy_or(app):
    from subdisc.data_loader_txt import DataLoaderTXT
    from subdisc.gui.browse_window import BrowseTableModel
    t = DataLoaderTXT(resource("adult.txt")).get_table()
    m = BrowseTableModel(t)
    m.set_filter(0, "<", "20", "or", ">", "80")   # age
    ages = t.get_column(0).floats_view()[:t.get_nr_rows()]
    assert m.rowCount() == int(np.count_nonzero((ages < 20) | (ages > 80)))
    m.set_filter(9, "=", "male")                   # and sex = Male
    assert all(m.display(m.source_row(i), 9) == "Male" for i in range(m.rowCount()))


def test_presets_save_apply_delete(app, presets_store):
    from subdisc.gui.table_filter import FilterTableView, ListTableModel, load_presets
    a = FilterTableView(ListTableModel(["Coverage", "Conditions"], [[100, "x"], [500, "y"]]), preset_kind="t")
    a.model().set_filter(0, ">=", "300")
    assert a.save_preset("big ones")
    assert "big ones" in load_presets("t")
    # another table of the same kind, columns in another order
    b = FilterTableView(ListTableModel(["Conditions", "coverage", "Other"], [["x", 100, 1], ["y", 500, 2]]), preset_kind="t")
    assert b.apply_preset("big ones", quiet=True) == []
    assert b.model().rowCount() == 1 and b.model().display(b.model().source_row(0), 0) == "y"
    # a table without that column reports it
    c = FilterTableView(ListTableModel(["Quality"], [[1.0]]), preset_kind="t")
    assert c.apply_preset("big ones", quiet=True) == ["coverage"]
    a.delete_preset("big ones")
    assert "big ones" not in load_presets("t")


def test_presets_from_other_kind(app, presets_store):
    from subdisc.gui.table_filter import FilterTableView, ListTableModel, preset_kinds
    a = FilterTableView(ListTableModel(["age", "sex"], [[30, "Male"], [60, "Female"]]), preset_kind="browse")
    a.model().set_filter(0, ">", "50")
    a.save_preset("older")
    assert preset_kinds() == ["browse"]
    b = FilterTableView(ListTableModel(["Attribute v / Target >", "age"], [["x", 10], ["y", 70]]), preset_kind="cross")
    assert b.apply_preset("older", quiet=True) is None            # not in its own kind
    assert b.apply_preset("older", quiet=True, kind="browse") == []
    assert b.model().rowCount() == 1


def test_filter_info_line(app):
    from subdisc.gui.table_filter import FilterInfo, FilterTableView, ListTableModel
    v = FilterTableView(ListTableModel(["n"], [[1], [2], [3]]))
    info = FilterInfo(v, "numbers", "extra words")
    assert info.isHidden()
    v.model().set_filter(0, ">", "1")
    assert not info.isHidden()
    assert "Showing 2 of 3 numbers" in info.text() and "extra words" in info.text()
    info.linkActivated.emit("clear")                 # the 'clear filters' link
    assert not v.model().is_filtered() and info.isHidden()


def test_browse_numeric_include_same_as_row_by_row(app):
    from subdisc.data_loader_txt import DataLoaderTXT
    from subdisc.gui.browse_window import BrowseTableModel
    from subdisc.gui.table_filter import match
    t = DataLoaderTXT(resource("adult.txt")).get_table()
    m = BrowseTableModel(t)
    for op, text in (("include", "4"), ("exclude", "0"), ("include", "1,0")):
        m.set_filter(2, op, text)                    # fnlwgt, shown with a grouping comma
        got = set(int(r) for r in m.visible_source_rows())
        want = {r for r in range(t.get_nr_rows())
                if match(float(t.get_column(2).get_float(r)), m.display(r, 2), op, text)}
        assert got == want
    assert m.distinct_texts(0)[:2] == ["17", "18"]   # age, numeric completer values


def test_roc_table_filter_redraws_curve(app):
    from subdisc import process
    from subdisc.file_handler import Action, FileHandler
    from subdisc.gui.mining_window import MiningWindow
    from subdisc.gui.roc_curve_window import ROCCurveWindow
    t = FileHandler(Action.OPEN_FILE, resource("adult.txt")).get_table()
    w = MiningWindow(t)
    w._setup_search_parameters()
    sd = process.run_subgroup_discovery(t, 0, None, w._search_parameters, False, 1, None)
    roc = ROCCurveWindow(sd.get_result(), w._search_parameters, sd.get_quality_measure())
    n_all = len(roc._curve._points)
    roc._model.set_filter(3, "include", "age")      # Conditions
    shown = [roc._model.display(roc._model.source_row(i), 3) for i in range(roc._model.rowCount())]
    assert shown and all("age" in s for s in shown)
    assert 0 < len(roc._curve._points) < n_all
    assert "table filter" in roc.windowTitle()
    roc._model.clear_filters()
    assert len(roc._curve._points) == n_all
    roc.close()
    w.hide()
