"""Secondary / tertiary regression targets (Cook's distance). The button that
opens this window is hidden in the java stable jar, and here as well."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QHBoxLayout, QLabel, QRadioButton,
                               QVBoxLayout)

from .. import column as column_module
from .. import log
from ..attribute_type import AttributeType
from ..column import Column
from . import common
from .table_filter import FilterInfo, FilterSortModel, FilterTableView

HEADER = ("Attribute", "Target status", "Type")


class MultiRegressionTargetsTableModel(FilterSortModel):
    def __init__(self, table):
        super().__init__()
        self._table = table
        self.rebuild()

    def refresh(self):
        self.rebuild()

    def source_rows(self):
        return range(self._table.get_nr_columns())

    def column_count(self):
        return len(HEADER)

    def header(self, col):
        return HEADER[col]

    def display(self, src, col):
        c = self._table.get_column(src)
        if col == 0:
            return c.get_name()
        if col == 1:
            return c.display_target_status()
        return str(c.get_type())


class MultiRegressionTargetsWindow(common.Window):
    def __init__(self, search_parameters, table, mining_window):
        super().__init__()
        self._sp = search_parameters
        self._table = table
        self._mw = mining_window
        if search_parameters is None or table is None or mining_window is None:
            log.log_command_line("MultiRegressionTargetsWindow Constructor: parameters can not be 'null'.")
            return
        self._model = MultiRegressionTargetsTableModel(table)
        self._view = FilterTableView(self._model, preset_kind="regression")
        self._view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._view.verticalHeader().setVisible(False)
        self._view.setColumnWidth(0, int(0.3 * common.WINDOW_DEFAULT_SIZE.width()))
        self._view.fit_headings()
        self._view.horizontalHeader().setStretchLastSection(True)
        self.content().addWidget(self._view, 1)
        self.content().addWidget(FilterInfo(self._view, "attributes"))

        row = QHBoxLayout()
        st = common.build_border("Set Target Status")
        sl = QVBoxLayout(st)
        self._status = QButtonGroup(self)
        for i in range(column_module.FIRST_TARGET_STATUS, column_module.LAST_TARGET_STATUS + 1):
            text = Column.get_target_text(i)
            rb = QRadioButton(text.lower())
            rb.setProperty("status", text)
            self._status.addButton(rb)
            sl.addWidget(rb)
            if i == column_module.FIRST_TARGET_STATUS:
                rb.setChecked(True)
        sl.addWidget(common.build_button("Change Target Status", "C", self._change))
        row.addWidget(st)
        ip = common.build_border("Include intercept in validation")
        il = QVBoxLayout(ip)
        il.addWidget(QLabel("Include intercept"))
        self._intercept = common.build_combo_box(["No", "Yes"])
        self._intercept.setCurrentIndex(1)
        il.addWidget(self._intercept)
        row.addWidget(ip)
        self.content().addLayout(row)
        self.add_button_row([common.build_button("Close", "C", self._closing_hook)])
        fb = QHBoxLayout()
        fb.addWidget(QLabel(" Last Action: "))
        fb.addWidget(QLabel("Meta Data loaded for " + table.get_name()), 1)
        self.content().addLayout(fb)
        self.setWindowTitle("Setting secondary and tertiary regression variables for: " + table.get_name())
        self.show_at(100, 100, common.WINDOW_DEFAULT_SIZE)

    def _change(self):
        status = self._status.checkedButton().property("status")
        for i in sorted(self._view.selected_source_rows()):
            self._table.get_column(i).set_target_status(status)
        self._model.refresh()
        self._mw.update_table()

    def _closing_hook(self):
        secondary, tertiary = [], []
        tc = self._sp.get_target_concept()
        nr_primary = nr_secondary = 0
        for c in self._table.get_columns():
            if c.get_type() == AttributeType.NUMERIC:
                s = c.get_target_status()
                if s == column_module.PRIMARY:
                    nr_primary += 1
                    tc.set_primary_target(c)
                elif s == column_module.SECONDARY:
                    nr_secondary += 1
                    secondary.append(c)
                elif s == column_module.TERTIARY:
                    tertiary.append(c)
        tc.set_secondary_targets(secondary)
        tc.set_tertiary_targets(tertiary)
        tc.set_intercept_relevance(self._intercept.currentIndex() == 1)
        if nr_primary == 1 and nr_secondary > 0:
            self.close()
