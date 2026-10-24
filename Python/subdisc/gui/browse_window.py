"""BrowseWindow: all data of a Table (or the members of one subgroup).

The header shows the column name and its number of distinct values. For a
nominal target the rows of a subgroup that are not positives get a red
background. Clicking a header sorts on that column, the arrow at its right
end filters (see table_filter.py).
"""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QFontMetrics
from PySide6.QtWidgets import QAbstractItemView, QComboBox

from .. import log
from ..attribute_type import AttributeType
from . import common
from .table_filter import FilterInfo, FilterSortModel, FilterTableView, match, parse_number, sort_key


class BrowseTableModel(FilterSortModel):
    def __init__(self, table, members=None, true_positives=None):
        super().__init__()
        self._table = table
        self._members = members
        self._tp = true_positives
        self._red = QBrush(common.RED)
        self.rebuild()

    def source_rows(self):
        n = self._table.get_nr_rows()
        if self._members is None:
            return np.arange(n)
        return np.flatnonzero(np.asarray(self._members[:n], dtype=bool))

    def column_count(self):
        return self._table.get_nr_columns()

    def header(self, col):
        c = self._table.get_column(col)
        return "%s\n(%d distinct)" % (c.get_name(), c.get_cardinality())

    def vertical_header(self, src):
        return str(int(src) + 1)

    def display(self, src, col):
        c = self._table.get_column(col)
        row = int(src)
        t = c.get_type()
        if t == AttributeType.NOMINAL:
            return c.get_nominal(row)
        if t in (AttributeType.NUMERIC, AttributeType.ORDINAL):
            return common.number(c.get_float(row))
        if t == AttributeType.BINARY:
            if c.get_missing(row):
                return "?"
            return "1" if c.get_binary(row) else "0"
        log.log_command_line("BrowseTableModel.getValueAt(%d, %d), Unknown AttributeType: %s" % (row, col, t))
        return None

    def raw(self, src, col):
        c = self._table.get_column(col)
        row = int(src)
        t = c.get_type()
        if t in (AttributeType.NUMERIC, AttributeType.ORDINAL):
            return float(c.get_float(row))
        if t == AttributeType.BINARY:
            return None if c.get_missing(row) else (1 if c.get_binary(row) else 0)
        return c.get_nominal(row)

    def cell_role(self, src, col, role):
        if role == Qt.TextAlignmentRole:
            if self._table.get_column(col).get_type() == AttributeType.NUMERIC:
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return None
        if role == Qt.BackgroundRole:
            if self._tp is not None and not self._tp[int(src)]:
                return self._red
        return None

    # numpy versions, the data can have many rows
    def _states(self, c, rows):
        """for nominal and binary columns: a small code per row plus, per
        code, its (raw, display) pair"""
        n = c.size()
        if c.get_type() == AttributeType.NOMINAL:
            codes = c.nominal_codes_view()[:n][rows]
            return codes, [(v, v) for v in c.distinct_values()]
        b = c.binaries_view()[:n][rows].astype(np.int64)
        miss = c.missing_view()[:n][rows]
        return np.where(miss, 2, b), [(0, "0"), (1, "1"), (None, "?")]

    def filter_rows(self, rows, col, op, text):
        rows = np.asarray(rows, dtype=np.int64)
        c = self._table.get_column(col)
        if c.get_type() in (AttributeType.NOMINAL, AttributeType.BINARY):
            codes, states = self._states(c, rows)
            if not states:
                return rows[:0]
            lut = np.array([match(r, d, op, text) for r, d in states], dtype=bool)
            return rows[lut[codes]]
        n = c.size()
        vals = c.floats_view()[:n][rows].astype(np.float64)
        tn = parse_number(text)
        if op in ("include", "exclude") or tn is None:
            # compares the text as shown: format every distinct number one
            # time, not every row (a column has far fewer values than rows)
            if not rows.size:
                return rows
            uniq, inv = np.unique(vals, return_inverse=True)
            lut = np.array([match(float(u), common.number(u), op, text) for u in uniq], dtype=bool)
            return rows[lut[inv.reshape(-1)]]
        with np.errstate(invalid="ignore"):
            if op == "=":
                keep = np.abs(vals - tn) < 5e-7   # as shown, 6 decimals
            elif op == ">":
                keep = vals > tn
            elif op == "<":
                keep = vals < tn
            elif op == ">=":
                keep = vals >= tn
            else:
                keep = vals <= tn
        return rows[keep]

    def sort_rows(self, rows, col, descending):
        rows = np.asarray(rows, dtype=np.int64)
        c = self._table.get_column(col)
        if c.get_type() in (AttributeType.NOMINAL, AttributeType.BINARY):
            codes, states = self._states(c, rows)
            if not states:
                return rows
            # rank of every state, text ignoring case, '?' gets NaN (last)
            order = sorted(range(len(states)), key=lambda i: sort_key(states[i][0]))
            rank = np.empty(len(states), dtype=np.float64)
            for k, i in enumerate(order):
                rank[i] = np.nan if sort_key(states[i][0])[0] == 2 else k
            keys = rank[codes]
        else:
            n = c.size()
            keys = c.floats_view()[:n][rows].astype(np.float64)
        empty = np.isnan(keys)
        valid = np.flatnonzero(~empty)
        if descending:
            # equal keys keep their original order
            idx = valid[np.argsort(-keys[valid], kind="stable")]
        else:
            idx = valid[np.argsort(keys[valid], kind="stable")]
        return np.concatenate([rows[idx], rows[empty]])

    def distinct_texts(self, col, limit=500):
        c = self._table.get_column(col)
        if c.get_type() == AttributeType.NOMINAL:
            return sorted(c.get_domain(), key=str.lower)[:limit]
        if c.get_type() == AttributeType.BINARY:
            return ["0", "1", "?"]
        # numeric: the distinct values, when there are not too many
        n = c.size()
        uniq = np.unique(c.floats_view()[:n])
        if uniq.size > limit:
            return []
        return [common.number(u) for u in uniq if u == u]

    def visible_mask(self):
        """the rows shown, as a mask over the table"""
        m = np.zeros(self._table.get_nr_rows(), dtype=bool)
        m[np.asarray(self._rows, dtype=np.int64)] = True
        return m


class BrowseJTable(FilterTableView):
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
        self.preset_kind = "browse"
        self.setModel(BrowseTableModel(table, members, tp))
        self.setSelectionBehavior(QAbstractItemView.SelectColumns)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.verticalHeader().setDefaultSectionSize(22)
        self.horizontalHeader().setMinimumHeight(40)
        fm = QFontMetrics(self.horizontalHeader().font())
        w91 = fm.horizontalAdvance("(999 distinct)") + 16
        for i in range(table.get_nr_columns()):
            w = fm.horizontalAdvance(table.get_column(i).get_name()) + 16
            self.setColumnWidth(i, max(w, w91))
        self.fit_headings()   # room for the sort and filter arrows

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
        self.content().addWidget(FilterInfo(self._jtable, "rows", "Save writes only these."))

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
        m = self._jtable.model()
        # with a filter on, only the rows shown are saved
        self._table.to_file(m.visible_mask() if m.is_filtered() else self._members)

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
