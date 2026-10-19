"""MetaDataWindow: per column the cardinality, type, enabled and missing
values, with buttons to change type and to enable / disable columns.

TODO (java) if changes are made also update other opened windows, eg.
BrowseWindow
"""

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, QItemSelectionModel
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QHBoxLayout,
                               QLabel, QRadioButton, QTableView, QVBoxLayout)

from .. import log
from ..attribute_type import AttributeType
from . import common

HEADER = ("Attribute", "Cardinality", "Type", "Enabled", "Missing Values")


class MetaDataTableModel(QAbstractTableModel):
    def __init__(self, table):
        super().__init__()
        self._table = table

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._table.get_nr_columns()

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(HEADER)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if 0 <= section < len(HEADER):
                return HEADER[section]
            log.log_command_line("Error in MetaDataTableHeader.getColumnName(): invalid index '%d'." % section)
            return "Incorrect column index."
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        c = self._table.get_column(index.row())
        col = index.column()
        if col == 0:
            return c.get_name()
        if col == 1:
            return str(c.get_cardinality())
        if col == 2:
            return str(c.get_type())
        if col == 3:
            return "yes" if c.get_is_enabled() else "no"
        if col == 4:
            return "yes" if c.get_has_missing_values() else "no"
        return None

    def refresh(self):
        self.beginResetModel()
        self.endResetModel()


class MetaDataWindow(common.Window):
    def __init__(self, mining_window, table):
        super().__init__()
        self._mining_window = mining_window
        self._table = table
        if table is None or mining_window is None:
            log.log_command_line("MetaDataWindow Constructor: parameters can not be 'null'.")
            return
        self._model = MetaDataTableModel(table)
        self._view = QTableView()
        self._view.setModel(self._model)
        self._view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._view.setAlternatingRowColors(True)
        self._view.verticalHeader().setVisible(False)
        self._view.verticalHeader().setDefaultSectionSize(22)
        w = common.WINDOW_DEFAULT_SIZE.width()
        self._view.setColumnWidth(0, int(0.3 * w))
        for i in range(1, 5):
            self._view.setColumnWidth(i, int(0.15 * w))
        self._view.horizontalHeader().setStretchLastSection(True)
        self.content().addWidget(self._view, 1)

        actions = QHBoxLayout()
        # selection
        sel = common.build_border("Select")
        sl = QVBoxLayout(sel)
        sl.addWidget(common.build_button("All", "A", self._view.selectAll))
        sl.addWidget(common.build_button("All Nominal", "N", lambda: self._select_type(AttributeType.NOMINAL)))
        sl.addWidget(common.build_button("All Numeric", "U", lambda: self._select_type(AttributeType.NUMERIC)))
        sl.addWidget(common.build_button("All Binary", "B", lambda: self._select_type(AttributeType.BINARY)))
        sl.addWidget(common.build_button("Clear Selection", "X", self._view.clearSelection))
        actions.addWidget(sel)
        # change type, ORDINAL is left out until it is implemented
        tp = common.build_border("Set Type")
        tl = QVBoxLayout(tp)
        tl.addStretch(1)
        self._types = QButtonGroup(self)
        first = True
        for at in AttributeType:
            if at == AttributeType.ORDINAL:
                continue
            rb = QRadioButton(str(at).lower())
            rb.setProperty("type", at.name)
            self._types.addButton(rb)
            tl.addWidget(rb)
            if first:
                rb.setChecked(True)
                first = False
        tl.addStretch(1)
        tl.addWidget(common.build_button("Change Type", "C", self._change_type))
        actions.addWidget(tp)
        # disable / enable
        de = common.build_border("Set Disabled/Enabled")
        dl = QVBoxLayout(de)
        dl.addStretch(1)
        dl.addWidget(common.build_button("Disable Selected", "D", lambda: self._set_enabled(False)))
        dl.addStretch(1)
        dl.addWidget(common.build_button("Enable Selected", "E", lambda: self._set_enabled(True)))
        dl.addStretch(1)
        dl.addWidget(common.build_button("Toggle Selected", "T", self._toggle))
        actions.addWidget(de)
        self.content().addLayout(actions)

        close = common.build_button("Close", "C", self.close)
        self.add_button_row([close])
        fb = QHBoxLayout()
        fb.addWidget(QLabel(" Last Action: "))
        self._feedback = QLabel("Meta Data loaded for " + table.get_name())
        fb.addWidget(self._feedback, 1)
        self.content().addLayout(fb)
        close.setFocus()

        self.setWindowTitle("Meta Data for: " + table.get_name())
        self.show_at(100, 100, common.WINDOW_DEFAULT_SIZE)

    def _selected_rows(self):
        return sorted({i.row() for i in self._view.selectionModel().selectedRows()})

    def _select_type(self, at):
        sm = self._view.selectionModel()
        for i in range(self._table.get_nr_columns()):
            if self._table.get_column(i).get_type() == at:
                sm.select(self._model.index(i, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)

    def _after_change(self, text):
        # repaint, keep the selection
        rows = self._selected_rows()
        self._model.refresh()
        sm = self._view.selectionModel()
        for r in rows:
            sm.select(self._model.index(r, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
        self._feedback.setText(text)
        self._mining_window.update_table()

    def _set_enabled(self, enable):
        rows = self._selected_rows()
        for i in rows:
            self._table.get_column(i).set_is_enabled(enable)
        self._after_change("%s %d attribute%s" % ("Enabled" if enable else "Disabled", len(rows), "" if len(rows) == 1 else "s"))

    def _toggle(self):
        rows = self._selected_rows()
        for i in rows:
            c = self._table.get_column(i)
            c.set_is_enabled(not c.get_is_enabled())
        self._after_change("Toggled %d attribute%s" % (len(rows), "" if len(rows) == 1 else "s"))

    def _change_type(self):
        b = self._types.checkedButton()
        at = AttributeType[b.property("type")]
        rows = self._selected_rows()
        failed = 0
        for i in rows:
            # TODO (java) ask whether the first value is 'true' or 'false'
            if self._table.get_column(i).set_type(at) is False:
                failed += 1
        msg = "Changed type of %d attribute%s to %s" % (len(rows) - failed, "" if len(rows) - failed == 1 else "s", str(at))
        if failed:
            msg += " (%d failed)" % failed
        self._after_change(msg)
