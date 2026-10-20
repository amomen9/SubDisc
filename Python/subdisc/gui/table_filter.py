"""Sorting and filtering for every table with column headings (not in java).

- click a heading: sort on that column, ascending, then descending, then back
  to the original order
- the small arrow at the right end of a heading opens a filter: a value and
  one of =, >, <, >=, <=, include (the cell text contains the value) and
  exclude (it does not). Filters on several columns must all hold

Numbers are compared as numbers (also '5,178' with a grouping comma), the
rest as text, ignoring upper / lower case. '=' compares with the number as
the table shows it, so typing what you see works.

The models keep a list of visible source rows, windows that start other
things from a table (ROC, Pattern Team, Save, ...) ask for those rows.
"""

import re

from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPolygon
from PySide6.QtWidgets import (QComboBox, QCompleter, QFrame, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QPushButton, QTableView, QToolTip, QVBoxLayout)

from ..java_compat import double_to_string

OPS = ("=", ">", "<", ">=", "<=", "include", "exclude")
ARROW_W = 16

_GROUPED = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d*)?$")


def parse_number(text):
    """the number in a cell or a filter box, None if it is no number"""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s in ("∞", "+∞"):
        return float("inf")
    if s == "-∞":
        return float("-inf")
    if _GROUPED.match(s):
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def match(raw, display, op, text):
    """does one cell pass the filter (op, text)"""
    t = (text or "").strip()
    d = "" if display is None else str(display)
    if op == "include":
        return t.lower() in d.lower()
    if op == "exclude":
        return t.lower() not in d.lower()
    tn = parse_number(t)
    rn = float(raw) if _is_number(raw) else parse_number(d)
    if tn is not None and rn is not None:
        if rn != rn:   # NaN, an empty cell
            return False
        if op == "=":
            dn = parse_number(d)
            return (dn if dn is not None else rn) == tn
        if op == ">":
            return rn > tn
        if op == "<":
            return rn < tn
        if op == ">=":
            return rn >= tn
        if op == "<=":
            return rn <= tn
        return True
    a, b = d.strip().lower(), t.lower()
    if op == "=":
        return a == b
    if op == ">":
        return a > b
    if op == "<":
        return a < b
    if op == ">=":
        return a >= b
    if op == "<=":
        return a <= b
    return True


def sort_key(raw):
    """(group, value): numbers, then text, empty / NaN cells always last"""
    if raw is None:
        return (2, 0)
    if _is_number(raw):
        f = float(raw)
        return (2, 0) if f != f else (0, f)
    s = str(raw).strip()
    if s == "":
        return (2, 0)
    return (1, s.lower())


class FilterSortModel(QAbstractTableModel):
    """Base model: subclasses give the cells, this class keeps the visible
    rows (filtered and sorted) as a list of source row numbers.

    Subclasses implement column_count(), header(col), source_rows() (all
    rows, in their own order), display(src, col) and optionally raw(src, col),
    cell_role(src, col, role) and vertical_header(src)."""

    filtersChanged = Signal()

    def __init__(self):
        super().__init__()
        self._filters = {}     # col -> (op, text), in the order they were set
        self._sort = None      # (col, order) or None for the original order
        self._rows = []
        self._all = []

    # ---- to implement ------------------------------------------------
    def column_count(self):
        raise NotImplementedError

    def header(self, col):
        raise NotImplementedError

    def source_rows(self):
        raise NotImplementedError

    def display(self, src, col):
        raise NotImplementedError

    def raw(self, src, col):
        return self.display(src, col)

    def cell_role(self, src, col, role):
        return None

    def vertical_header(self, src):
        return None

    # ---- overridable for speed (see the browse window) -----------------
    def filter_rows(self, rows, col, op, text):
        return [r for r in rows if match(self.raw(r, col), self.display(r, col), op, text)]

    def sort_rows(self, rows, col, descending):
        keyed = [(sort_key(self.raw(r, col)), r) for r in rows]
        valid = [x for x in keyed if x[0][0] != 2]
        empty = [x[1] for x in keyed if x[0][0] == 2]
        valid.sort(key=lambda x: x[0], reverse=descending)
        return [x[1] for x in valid] + empty

    def distinct_texts(self, col, limit=500):
        """values for the completer of the filter box"""
        seen = []
        got = set()
        for r in self._all:
            d = self.display(r, col)
            if d is None:
                continue
            d = str(d)
            if d not in got:
                got.add(d)
                seen.append(d)
                if len(seen) >= limit:
                    break
        return seen

    # ---- Qt model ------------------------------------------------------
    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else self.column_count()

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return self.header(section)
        if 0 <= section < len(self._rows):
            return self.vertical_header(self._rows[section])
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._rows):
            return None
        src = self._rows[index.row()]
        if role == Qt.DisplayRole:
            d = self.display(src, index.column())
            return None if d is None else str(d)
        return self.cell_role(src, index.column(), role)

    def sort(self, column, order=Qt.AscendingOrder):
        self._sort = None if column < 0 else (column, order)
        self._relayout()

    # ---- rows ----------------------------------------------------------
    def rebuild(self):
        """the source changed (rows deleted, types changed): keep filters
        and sorting, recompute the visible rows"""
        self.beginResetModel()
        rows = self.source_rows()
        # a list, or a numpy array for big tables (the browse window)
        self._all = list(rows) if isinstance(rows, range) else rows
        self._rows = self._compute()
        self.endResetModel()
        self.filtersChanged.emit()

    def _compute(self):
        rows = self._all
        for col, (op, text) in self._filters.items():
            if col < self.column_count():
                rows = self.filter_rows(rows, col, op, text)
        if self._sort is not None and self._sort[0] < self.column_count():
            rows = self.sort_rows(rows, self._sort[0], self._sort[1] == Qt.DescendingOrder)
        return rows

    def _relayout(self):
        """sort only: keep the selection on the same source rows"""
        self.layoutAboutToBeChanged.emit()
        old = self.persistentIndexList()
        old_src = [(int(self._rows[i.row()]) if i.row() < len(self._rows) else None, i.column()) for i in old]
        self._rows = self._compute()
        pos = {int(r): i for i, r in enumerate(self._rows)} if old else {}
        new = []
        for src, col in old_src:
            if src in pos:
                new.append(self.index(pos[src], col))
            else:
                new.append(QModelIndex())
        self.changePersistentIndexList(old, new)
        self.layoutChanged.emit()

    def set_filter(self, col, op, text):
        self._filters[col] = (op, text)
        self.rebuild()

    def clear_filter(self, col):
        if self._filters.pop(col, None) is not None:
            self.rebuild()

    def clear_filters(self):
        if self._filters:
            self._filters.clear()
            self.rebuild()

    def filters(self):
        return dict(self._filters)

    def filter_of(self, col):
        return self._filters.get(col)

    def is_filtered(self):
        return bool(self._filters)

    def describe_filters(self):
        return "; ".join("%s %s %s" % (self.header(c).replace("\n", " "), op, text)
                         for c, (op, text) in self._filters.items())

    def source_row(self, view_row):
        return self._rows[view_row]

    def visible_source_rows(self):
        return list(self._rows)

    def all_source_rows(self):
        return list(self._all)

    def view_row(self, src):
        for i, r in enumerate(self._rows):
            if r == src:
                return i
        return -1


class FilterPopup(QFrame):
    def __init__(self, header, col):
        super().__init__(header, Qt.Popup)
        self.setFrameShape(QFrame.StyledPanel)
        self._header = header
        self._col = col
        model = header.model()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        name = model.header(col).replace("\n", " ")
        lay.addWidget(QLabel("Filter <b>%s</b>" % _html(name)))
        row = QHBoxLayout()
        self._op = QComboBox()
        self._op.addItems(OPS)
        self._value = QLineEdit()
        self._value.setPlaceholderText("value")
        self._value.setMinimumWidth(170)
        current = model.filter_of(col)
        if current is not None:
            self._op.setCurrentText(current[0])
            self._value.setText(current[1])
        try:
            comp = QCompleter(model.distinct_texts(col), self)
            comp.setCaseSensitivity(Qt.CaseInsensitive)
            comp.setFilterMode(Qt.MatchContains)
            self._value.setCompleter(comp)
        except Exception:
            pass
        self._value.returnPressed.connect(self._apply)
        row.addWidget(self._op)
        row.addWidget(self._value, 1)
        lay.addLayout(row)
        buttons = QHBoxLayout()
        apply = QPushButton("Apply")
        apply.setDefault(True)
        apply.clicked.connect(self._apply)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        clear_all = QPushButton("Clear all")
        clear_all.clicked.connect(self._clear_all)
        clear_all.setEnabled(model.is_filtered())
        buttons.addWidget(apply)
        buttons.addWidget(clear)
        buttons.addWidget(clear_all)
        lay.addLayout(buttons)
        self._value.setFocus()

    def _apply(self):
        self._header.model().set_filter(self._col, self._op.currentText(), self._value.text())
        self.close()

    def _clear(self):
        self._header.model().clear_filter(self._col)
        self.close()

    def _clear_all(self):
        self._header.model().clear_filters()
        self.close()


def _html(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class FilterHeader(QHeaderView):
    """horizontal header: click sorts, the arrow at the right end filters"""

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setSectionsClickable(True)
        self.setSortIndicatorShown(True)
        self.setSortIndicatorClearable(True)
        self.setSortIndicator(-1, Qt.AscendingOrder)
        self.setHighlightSections(False)
        self.setMouseTracking(True)
        self._hover = -1
        self._arrow_press = False
        self.sortIndicatorChanged.connect(self._sort_changed)

    def _sort_changed(self, section, order):
        m = self.model()
        if isinstance(m, FilterSortModel):
            m.sort(section, order)

    def arrow_rect(self, logical):
        x = self.sectionViewportPosition(logical)
        w = self.sectionSize(logical)
        return QRect(x + w - ARROW_W - 3, 0, ARROW_W, self.height())

    def paintSection(self, painter, rect, logical):
        painter.save()
        super().paintSection(painter, rect, logical)
        painter.restore()
        m = self.model()
        if not isinstance(m, FilterSortModel) or rect.width() < ARROW_W + 12:
            return
        active = m.filter_of(logical) is not None
        r = QRect(rect.right() - ARROW_W - 2, rect.top() + 2, ARROW_W, rect.height() - 4)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        accent = self.palette().color(self.palette().ColorRole.Highlight)
        if active or logical == self._hover:
            bg = QColor(accent)
            bg.setAlpha(60 if active else 30)
            painter.setPen(Qt.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(r, 3, 3)
        color = accent if active else self.palette().color(self.palette().ColorRole.WindowText)
        if not active:
            color.setAlpha(150)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        cx, cy = r.center().x(), r.center().y()
        painter.drawPolygon(QPolygon([QPoint(cx - 4, cy - 2), QPoint(cx + 4, cy - 2), QPoint(cx, cy + 3)]))
        if active:
            painter.drawRect(cx - 4, cy + 5, 8, 1)
        # the sort direction, left of the filter arrow (the style's own
        # indicator is switched off in theme.py, it would sit on top of it)
        if self.isSortIndicatorShown() and self.sortIndicatorSection() == logical:
            sc = self.palette().color(self.palette().ColorRole.WindowText)
            painter.setBrush(sc)
            sx = r.left() - 7
            if self.sortIndicatorOrder() == Qt.AscendingOrder:
                tri = [QPoint(sx - 4, cy + 2), QPoint(sx + 4, cy + 2), QPoint(sx, cy - 3)]
            else:
                tri = [QPoint(sx - 4, cy - 2), QPoint(sx + 4, cy - 2), QPoint(sx, cy + 3)]
            painter.drawPolygon(QPolygon(tri))
        painter.restore()

    def _arrow_at(self, pos):
        logical = self.logicalIndexAt(pos)
        if logical < 0:
            return -1
        return logical if self.arrow_rect(logical).contains(pos) else -1

    def mousePressEvent(self, e):
        logical = self._arrow_at(e.position().toPoint())
        if logical >= 0 and e.button() == Qt.LeftButton and isinstance(self.model(), FilterSortModel):
            self._arrow_press = True
            r = self.arrow_rect(logical)
            popup = FilterPopup(self, logical)
            popup.adjustSize()
            pt = self.viewport().mapToGlobal(QPoint(r.right() - popup.width(), r.bottom()))
            popup.move(pt)
            popup.show()
            e.accept()
            return
        self._arrow_press = False
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if self._arrow_press:
            self._arrow_press = False
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def mouseMoveEvent(self, e):
        h = self._arrow_at(e.position().toPoint())
        if h != self._hover:
            self._hover = h
            self.viewport().update()
        super().mouseMoveEvent(e)

    def leaveEvent(self, e):
        if self._hover != -1:
            self._hover = -1
            self.viewport().update()
        super().leaveEvent(e)

    def viewportEvent(self, e):
        if e.type() == QEvent.ToolTip:
            pos = e.pos()
            logical = self.logicalIndexAt(pos)
            m = self.model()
            if logical >= 0 and isinstance(m, FilterSortModel):
                f = m.filter_of(logical)
                if self.arrow_rect(logical).contains(pos):
                    tip = "Filter: %s %s" % f if f else "Filter this column"
                else:
                    tip = "Click to sort (ascending, descending, original order)"
                    if f:
                        tip += "\nFiltered: %s %s" % f
                QToolTip.showText(e.globalPos(), tip, self)
                return True
        return super().viewportEvent(e)


class FilterTableView(QTableView):
    """QTableView with the sorting / filtering header"""

    def __init__(self, model=None, parent=None):
        super().__init__(parent)
        self.setHorizontalHeader(FilterHeader(self))
        if model is not None:
            self.setModel(model)

    def setModel(self, model):
        super().setModel(model)
        h = self.horizontalHeader()
        h.setModel(model)
        # QTableView turns the indicator off when sorting is not 'enabled'
        # on the view; the header does the sorting itself here
        h.setSortIndicatorShown(True)

    def selected_source_rows(self):
        m = self.model()
        rows = sorted({i.row() for i in self.selectionModel().selectedRows()})
        return [m.source_row(r) for r in rows]

    def select_first(self):
        if self.model().rowCount() > 0:
            self.selectRow(0)

    def fit_headings(self, extra=48):
        """widen columns whose heading (plus the sort and filter arrows)
        does not fit"""
        fm = self.horizontalHeader().fontMetrics()
        m = self.model()
        for c in range(m.columnCount()):
            text = m.headerData(c, Qt.Horizontal) or ""
            need = max(fm.horizontalAdvance(line) for line in str(text).split("\n")) + extra
            if self.columnWidth(c) < need:
                self.setColumnWidth(c, need)


class ListTableModel(FilterSortModel):
    """a plain list of rows (tuples) with headings"""

    def __init__(self, headers, rows, right_aligned=()):
        super().__init__()
        self._headers = list(headers)
        self._data = [tuple(r) for r in rows]
        self._right = set(right_aligned)
        self.rebuild()

    def column_count(self):
        return len(self._headers)

    def header(self, col):
        return str(self._headers[col])

    def source_rows(self):
        return range(len(self._data))

    def raw(self, src, col):
        return self._data[src][col]

    def display(self, src, col):
        v = self._data[src][col]
        if isinstance(v, float):
            return double_to_string(v)
        return "" if v is None else str(v)

    def cell_role(self, src, col, role):
        if role == Qt.TextAlignmentRole and (col in self._right or _is_number(self._data[src][col])):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None
