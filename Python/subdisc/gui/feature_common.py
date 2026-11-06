"""Small pieces shared by the windows of the new features (subgroup list,
holdout test, stability, multiple testing, model errors): a settings
dialog, a table window with the usual sorting / filtering, and the 'which
test data?' question. Not in the java version."""

import math

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QMessageBox, QRadioButton, QSpinBox,
                               QVBoxLayout)

from .. import ui_hooks
from ..file_type import FileType
from . import common
from .table_filter import FilterInfo, FilterTableView, ListTableModel


def nice(x, digits=4):
    """floats rounded for the tables (sorting still works on the number)"""
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return x
        if x != 0 and abs(x) < 10 ** -digits:
            return float("%.*g" % (digits, x))
        return round(x, digits)
    return x


def html_table(pairs):
    """a dict / list of (name, value) as a two column html table"""
    items = pairs.items() if isinstance(pairs, dict) else pairs
    cells = []
    for k, v in items:
        if isinstance(v, float):
            v = "-" if math.isnan(v) else ("%.4g" % v if abs(v) < 1e-3 and v != 0 else "{:,.4f}".format(v).rstrip("0").rstrip("."))
        elif isinstance(v, int):
            v = "{:,}".format(v)
        cells.append("<tr><td>%s</td><td align='right'>&nbsp;&nbsp;%s</td></tr>" % (_esc(str(k)), _esc(str(v))))
    return "<table cellspacing='0' cellpadding='1'>%s</table>" % "".join(cells)


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SettingsDialog(QDialog):
    """a form of a few fields, values() gives a dict or None (cancelled)"""

    def __init__(self, parent, title, intro=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowIcon(common.icon())
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._lay = QVBoxLayout(self)
        if intro:
            lbl = QLabel(intro)
            lbl.setWordWrap(True)
            lbl.setMaximumWidth(460)
            self._lay.addWidget(lbl)
        self._form = QFormLayout()
        self._lay.addLayout(self._form)
        self._getters = {}
        self._result = None

    def add_int(self, key, label, default, lo, hi):
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setValue(default)
        self._form.addRow(label, w)
        self._getters[key] = w.value
        return w

    def add_float(self, key, label, default, lo, hi, decimals=3, step=0.05):
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setDecimals(decimals)
        w.setSingleStep(step)
        w.setValue(default)
        self._form.addRow(label, w)
        self._getters[key] = w.value
        return w

    def add_combo(self, key, label, items, current=None):
        w = QComboBox()
        for it in items:
            w.addItem(str(it))
        if current is not None and w.findText(str(current)) >= 0:
            w.setCurrentText(str(current))
        self._form.addRow(label, w)
        self._getters[key] = w.currentText
        return w

    def add_check(self, key, label, checked=False):
        w = QCheckBox(label)
        w.setChecked(checked)
        self._form.addRow("", w)
        self._getters[key] = w.isChecked
        return w

    def add_note(self, text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: palette(mid);")
        self._form.addRow(lbl)
        return lbl

    def values(self):
        ok = common.build_button("OK", "O", self._ok, size=common.BUTTON_MEDIUM_SIZE)
        cancel = common.build_button("Cancel", "C", self.reject, size=common.BUTTON_MEDIUM_SIZE)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(ok)
        row.addWidget(cancel)
        self._lay.addLayout(row)
        ok.setDefault(True)
        self.exec()
        return self._result

    def _ok(self):
        self._result = {k: g() for k, g in self._getters.items()}
        self.accept()


class TableResultWindow(common.Window):
    """a summary line, a sortable / filterable table and Save / Close"""

    def __init__(self, title, headers, rows, summary_html="", noun="rows", preset_kind="table",
                 extra_buttons=(), size=None):
        super().__init__(title)
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._summary.setTextFormat(Qt.RichText)
        self._summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.content().addWidget(self._summary)
        self.set_summary(summary_html)
        self._model = ListTableModel(headers, [[nice(v) for v in r] for r in rows])
        self._view = FilterTableView(self._model, preset_kind=preset_kind)
        self._view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._view.setAlternatingRowColors(True)
        self._view.verticalHeader().setVisible(False)
        self._view.verticalHeader().setDefaultSectionSize(22)
        self._view.fit_headings()
        self._view.horizontalHeader().setStretchLastSection(True)
        self.content().addWidget(self._view, 1)
        self.content().addWidget(FilterInfo(self._view, noun))
        buttons = list(extra_buttons)
        buttons.append(common.build_button("Save...", "S", self._save, common.BUTTON_MEDIUM_SIZE))
        buttons.append(common.build_button("Close", "C", self.close, common.BUTTON_MEDIUM_SIZE))
        self.add_button_row(buttons)
        self.show_at(240, 160, size or common.WINDOW_DEFAULT_SIZE)

    def set_summary(self, html):
        self._summary.setText(html)
        self._summary.setVisible(bool(html))

    def set_rows(self, rows):
        self._model.set_rows([[nice(v) for v in r] for r in rows])

    def view(self):
        return self._view

    def model(self):
        return self._model

    def _save(self):
        path = ui_hooks.choose_save_file(FileType.ALL_DATA_FILES)
        if path is None:
            return
        m = self._model
        try:
            with open(path, "w", encoding="utf-8", newline="") as w:
                w.write("\t".join(m.header(c) for c in range(m.column_count())) + "\n")
                for r in m.visible_source_rows():
                    w.write("\t".join(str(m.display(r, c)) for c in range(m.column_count())) + "\n")
        except OSError as e:
            QMessageBox.warning(self, "Save", str(e))


class TestDataChooser(QDialog):
    """held out rows of the same table, or a test file"""

    def __init__(self, parent, table, selection, what="the subgroups"):
        super().__init__(parent)
        self.setWindowTitle("Test data")
        self.setWindowIcon(common.icon())
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._choice = None
        n = table.get_nr_rows()
        held = 0 if selection is None else n - int(np.count_nonzero(selection[:n]))
        lay = QVBoxLayout(self)
        lbl = QLabel("Which rows should %s be tested on? They are scored with the same quality measure, "
                     "nothing is mined again." % what)
        lbl.setWordWrap(True)
        lbl.setMaximumWidth(460)
        lay.addWidget(lbl)
        grp = QButtonGroup(self)
        self._rest = QRadioButton("Rows of '%s' that were not in the mining selection (%s rows)" % (table.get_name(), "{:,}".format(held)))
        self._file = QRadioButton("A test file with the same columns...")
        grp.addButton(self._rest)
        grp.addButton(self._file)
        lay.addWidget(self._rest)
        lay.addWidget(self._file)
        if held > 0:
            self._rest.setChecked(True)
        else:
            self._rest.setEnabled(False)
            self._file.setChecked(True)
            hint = QLabel("(Tools > Random Holdout Split in the main window keeps rows out of the mining run.)")
            hint.setStyleSheet("color: palette(mid);")
            hint.setWordWrap(True)
            lay.addWidget(hint)
        row = QHBoxLayout()
        row.addStretch(1)
        ok = common.build_button("OK", "O", self._ok, size=common.BUTTON_MEDIUM_SIZE)
        row.addWidget(ok)
        row.addWidget(common.build_button("Cancel", "C", self.reject, size=common.BUTTON_MEDIUM_SIZE))
        lay.addLayout(row)
        ok.setDefault(True)
        self._table = table
        self._selection = selection

    def _ok(self):
        self._choice = "rest" if self._rest.isChecked() else "file"
        self.accept()

    def choose(self):
        """('rest', table, rows) / ('file', test_table, None) / None"""
        from ..holdout import load_test_table
        self.exec()
        if self._choice is None:
            return None
        if self._choice == "rest":
            n = self._table.get_nr_rows()
            return "rest", self._table, ~np.asarray(self._selection[:n], dtype=bool)
        path = ui_hooks.choose_open_file(FileType.ALL_DATA_FILES)
        if not path:
            return None
        common.set_busy(True)
        try:
            t, warnings = load_test_table(self._table, path)
        except Exception as e:
            common.set_busy(False)
            QMessageBox.warning(self.parent(), "Test file", str(e))
            return None
        finally:
            common.set_busy(False)
        if warnings:
            QMessageBox.information(self.parent(), "Test file", "The file was loaded, but:\n- " + "\n- ".join(warnings))
        return "file", t, None
