"""BrowseWindow: all data of a Table (or the members of one subgroup).

The header shows the column name and its number of distinct values. For a
nominal target the rows of a subgroup that are not positives get a red
background. Clicking a header sorts on that column.
"""

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QBrush, QFontMetrics
from PySide6.QtWidgets import QAbstractItemView, QComboBox, QTableView

from .. import log
from ..attribute_type import AttributeType
from . import common


class BrowseTableModel(QAbstractTableModel):
    def __init__(self, table, members=None, true_positives=None):
        super().__init__()
        self._table = table
        self._members = members
        self._tp = true_positives
        n = table.get_nr_rows()
        if members is None:
            self._rows = np.arange(n)
        else:
            self._rows = np.flatnonzero(np.asarray(members[:n], dtype=bool))
        self._red = QBrush(common.RED)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else int(self._rows.shape[0])

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self._table.get_nr_columns()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            c = self._table.get_column(section)
            return "%s\n(%d distinct)" % (c.get_name(), c.get_cardinality())
        return str(int(self._rows[section]) + 1)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = int(self._rows[index.row()])
        if role == Qt.DisplayRole:
            c = self._table.get_column(index.column())
            t = c.get_type()
            if t == AttributeType.NOMINAL:
                return c.get_nominal(row)
            if t in (AttributeType.NUMERIC, AttributeType.ORDINAL):
                return common.number(c.get_float(row))
            if t == AttributeType.BINARY:
                if c.get_missing(row):
                    return "?"
                return "1" if c.get_binary(row) else "0"
            log.log_command_line("BrowseTableModel.getValueAt(%d, %d), Unknown AttributeType: %s" % (row, index.column(), t))
            return None
        if role == Qt.TextAlignmentRole:
            c = self._table.get_column(index.column())
            if c.get_type() == AttributeType.NUMERIC:
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return None
        if role == Qt.BackgroundRole:
            if self._tp is not None and not self._tp[row]:
                return self._red
            return None
        return None

    def sort(self, column, order=Qt.AscendingOrder):
        if column < 0:
            return
        c = self._table.get_column(column)
        n = self._table.get_nr_rows()
        t = c.get_type()
        if t == AttributeType.NOMINAL:
            keys = np.array([c.get_nominal(int(r)) for r in self._rows], dtype=object)
            idx = sorted(range(len(keys)), key=lambda i: keys[i])
            idx = np.asarray(idx, dtype=np.int64)
        elif t == AttributeType.BINARY:
            keys = c.binaries_view()[:n][self._rows].astype(np.int8)
            idx = np.argsort(keys, kind="stable")
        else:
            keys = c.floats_view()[:n][self._rows]
            idx = np.argsort(keys, kind="stable")
        if order == Qt.DescendingOrder:
            idx = idx[::-1]
        self.layoutAboutToBeChanged.emit()
        self._rows = self._rows[idx]
        self.layoutChanged.emit()


class BrowseJTable(QTableView):
    def __init__(self, table, subgroup=None):
        super().__init__()
        members = None
        tp = None
        if subgroup is not None:
            members = subgroup.get_members()
            tp = subgroup.get_parent_set().get_binary_target_clone()
            if tp is not None:
                n = min(len(tp), len(members))
                tp = np.asarray(tp[:n], dtype=bool) & np.asarray(members[:n], dtype=bool)
        self.setModel(BrowseTableModel(table, members, tp))
        self.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.setSortingEnabled(True)
        self.verticalHeader().setDefaultSectionSize(22)
        self.horizontalHeader().setMinimumHeight(40)
        fm = QFontMetrics(self.horizontalHeader().font())
        w91 = fm.horizontalAdvance("(999 distinct)") + 16
        for i in range(table.get_nr_columns()):
            w = fm.horizontalAdvance(table.get_column(i).get_name()) + 16
            self.setColumnWidth(i, max(w, w91))

    def focus_column(self, model_index):
        if model_index < 0 or model_index >= self.model().columnCount():
            self.clearSelection()
        else:
            self.selectColumn(model_index)
            if self.model().rowCount() > 0:
                self.scrollTo(self.model().index(0, model_index), QAbstractItemView.PositionAtCenter)


class BrowseWindow(common.Window):
    def __init__(self, table, subgroup=None):
        super().__init__()
        if table is None:
            log.log_command_line("BrowseWindow Constructor: theTable can not be 'null'.")
            return
        self._table = table
        self._members = None
        self._jtable = BrowseJTable(table, subgroup)
        self.content().addWidget(self._jtable, 1)

        self._map = {}
        for i in range(table.get_nr_columns()):
            c = table.get_column(i)
            self._map.setdefault(c.get_name(), c.get_index())
        self._box = QComboBox()
        self._box.setEditable(True)
        self._box.setInsertPolicy(QComboBox.NoInsert)
        self._box.addItems(list(self._map.keys()))
        self._box.setMinimumSize(common.BUTTON_DEFAULT_SIZE)
        self._box.activated.connect(lambda _i: self._update_columns_box())
        self._box.lineEdit().returnPressed.connect(self._update_columns_box)
        close = common.build_button("Close", "C", self.close)
        self.add_button_row([self._box, common.build_button("Save", "S", self._save), close])

        if subgroup is None:
            self._box.setCurrentIndex(0)
            self.setWindowTitle("Data for: " + table.get_name())
        else:
            self._members = subgroup.get_members()
            name = subgroup.get_conditions().get(0).get_column().get_name() if subgroup.get_depth() > 0 else self._box.itemText(0)
            self._box.setEditText(name)
            self._update_columns_box()
            self.setWindowTitle("%d members in subgroup: %s" % (subgroup.get_coverage(), subgroup.get_conditions()))
        close.setFocus()
        self.show_at(100, 100, common.WINDOW_DEFAULT_SIZE)

    def _save(self):
        self._table.to_file(self._members)

    def _update_columns_box(self):
        text = self._box.currentText()
        if text in self._map:
            self._jtable.focus_column(self._map[text])
            return
        low = text.lower()
        self._box.blockSignals(True)
        self._jtable.clearSelection()
        self._box.clear()
        # '' fits every item, resets the full list
        for k in self._map:
            if k.lower().startswith(low):
                self._box.addItem(k)
        if self._box.count() > 0:
            self._box.setCurrentIndex(0)
            self._jtable.focus_column(self._map[self._box.itemText(0)])
        else:
            self._box.setEditText(text)
        self._box.blockSignals(False)
