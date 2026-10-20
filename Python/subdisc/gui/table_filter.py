"""Sorting and filtering for every table with column headings (not in java).

- click a heading: sort on that column, ascending, then descending, then back
  to the original order
- the small arrow at the right end of a heading opens a filter: a value and
  one of =, >, <, >=, <=, include (the cell text contains the value) and
  exclude (it does not). A second condition on the same column can be added,
  joined with 'and' or 'or'. Filters on several columns must all hold
- the filters of a table can be saved as a named preset and applied again
  later, also in other windows of the same kind (presets menu in the filter
  box, or a right click on a heading). Presets are kept on this computer
  only (QSettings), the result and autorun files stay as java writes them

Numbers are compared as numbers (also '5,178' with a grouping comma), the
rest as text, ignoring upper / lower case. '=' compares with the number as
the table shows it, so typing what you see works.

The models keep a list of visible source rows, windows that start other
things from a table (ROC, Pattern Team, Save, ...) ask for those rows.
"""

import json
import re

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QPoint, QRect, QSettings, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPolygon
from PySide6.QtWidgets import (QComboBox, QCompleter, QFrame, QGridLayout, QHBoxLayout, QHeaderView,
                               QInputDialog, QLabel, QLineEdit, QMenu, QMessageBox, QPushButton,
                               QTableView, QToolTip, QVBoxLayout)

from ..java_compat import double_to_string

OPS = ("=", ">", "<", ">=", "<=", "include", "exclude")
JOINS = ("and", "or")
ARROW_W = 16
# where the presets live, tests point this somewhere else
PRESET_STORE = ("SubDisc", "SubDisc")

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
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def match(raw, display, op, text):
    """does one cell pass one condition (op, text)"""
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


class ColumnFilter:
    """one or two conditions on one column; a second condition only counts
    when its value box is not empty"""

    __slots__ = ("conds", "join")

    def __init__(self, op, text, join=None, op2=None, text2=None):
        self.conds = [(op, text or "")]
        self.join = "and"
        if join in JOINS and op2 in OPS and (text2 or "").strip():
            self.conds.append((op2, text2))
            self.join = join

    def accepts(self, raw, display):
        hits = [match(raw, display, op, t) for op, t in self.conds]
        return all(hits) if self.join == "and" else any(hits)

    def describe(self):
        s = "%s %s" % self.conds[0]
        if len(self.conds) > 1:
            s += " %s %s %s" % (self.join, self.conds[1][0], self.conds[1][1])
        return s

    def to_json(self):
        return {"conds": [list(c) for c in self.conds], "join": self.join}

    @staticmethod
    def from_json(d):
        conds = d.get("conds") or []
        if not conds:
            return None
        op, text = conds[0]
        if len(conds) > 1:
            return ColumnFilter(op, text, d.get("join", "and"), conds[1][0], conds[1][1])
        return ColumnFilter(op, text)

    def __eq__(self, other):
        return isinstance(other, ColumnFilter) and self.conds == other.conds and self.join == other.join

    def __repr__(self):
        return "ColumnFilter(%s)" % self.describe()


def heading_key(text):
    """presets match columns on the first line of the heading, ignoring case"""
    return str(text or "").split("\n")[0].strip().lower()


class FilterSortModel(QAbstractTableModel):
    """Base model: subclasses give the cells, this class keeps the visible
    rows (filtered and sorted) as a list of source row numbers.

    Subclasses implement column_count(), header(col), source_rows() (all
    rows, in their own order), display(src, col) and optionally raw(src, col),
    cell_role(src, col, role) and vertical_header(src)."""

    filtersChanged = Signal()

    def __init__(self):
        super().__init__()
        self._filters = {}     # col -> ColumnFilter, in the order they were set
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
        """the rows that pass one condition"""
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

    def _apply_column_filter(self, rows, col, cf):
        first = self.filter_rows(rows, col, *cf.conds[0])
        if len(cf.conds) == 1:
            return first
        if cf.join == "and":
            return self.filter_rows(first, col, *cf.conds[1])
        second = self.filter_rows(rows, col, *cf.conds[1])
        # 'or': keep the rows in their original order
        if isinstance(rows, np.ndarray):
            keep = np.isin(rows, np.asarray(first)) | np.isin(rows, np.asarray(second))
            return rows[keep]
        both = set(first) | set(second)
        return [r for r in rows if r in both]

    def _compute(self):
        rows = self._all
        for col, cf in self._filters.items():
            if col < self.column_count():
                rows = self._apply_column_filter(rows, col, cf)
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

    def set_filter(self, col, op, text, join=None, op2=None, text2=None):
        self._filters[col] = ColumnFilter(op, text, join, op2, text2)
        self.rebuild()

    def set_filters(self, filters):
        """replace all filters at once {col: ColumnFilter}"""
        self._filters = dict(filters)
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
        return "; ".join("%s %s" % (self.header(c).replace("\n", " "), cf.describe())
                         for c, cf in self._filters.items())

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

    # ---- presets -------------------------------------------------------
    def filters_as_preset(self):
        return {heading_key(self.header(c)): cf.to_json() for c, cf in self._filters.items()}

    def apply_preset(self, preset):
        """returns the headings of the preset that this table does not have"""
        by_key = {heading_key(self.header(c)): c for c in range(self.column_count())}
        filters, missing = {}, []
        for key, d in preset.items():
            cf = ColumnFilter.from_json(d)
            if cf is None:
                continue
            if key in by_key:
                filters[by_key[key]] = cf
            else:
                missing.append(key)
        self.set_filters(filters)
        return missing


# ----------------------------------------------------------------------
# presets on disk
# ----------------------------------------------------------------------

def _settings():
    # a path means an ini file (the tests use one), else (org, app)
    if isinstance(PRESET_STORE, str):
        return QSettings(PRESET_STORE, QSettings.IniFormat)
    return QSettings(*PRESET_STORE)


def load_presets(kind):
    try:
        s = _settings().value("filter_presets/" + kind, "")
        d = json.loads(s) if s else {}
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_presets(kind, presets):
    st = _settings()
    st.setValue("filter_presets/" + kind, json.dumps(presets))
    st.sync()


def preset_menu(view, parent=None):
    """the presets menu of a FilterTableView (filter box and heading right click)"""
    m = view.model()
    kind = view.preset_kind
    presets = load_presets(kind)
    menu = QMenu(parent or view)
    a = menu.addAction("Save filters as preset...")
    a.setEnabled(m.is_filtered())
    a.triggered.connect(lambda: view.save_preset())
    apply_menu = menu.addMenu("Apply preset")
    delete_menu = menu.addMenu("Delete preset")
    for name in sorted(presets, key=str.lower):
        apply_menu.addAction(name).triggered.connect(lambda _=False, n=name: view.apply_preset(n))
        delete_menu.addAction(name).triggered.connect(lambda _=False, n=name: view.delete_preset(n))
    apply_menu.setEnabled(bool(presets))
    delete_menu.setEnabled(bool(presets))
    menu.addSeparator()
    c = menu.addAction("Clear all filters")
    c.setEnabled(m.is_filtered())
    c.triggered.connect(m.clear_filters)
    return menu


# ----------------------------------------------------------------------
# widgets
# ----------------------------------------------------------------------

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
        try:
            values = model.distinct_texts(col)
        except Exception:
            values = []
        grid = QGridLayout()
        self._op = QComboBox()
        self._op.addItems(OPS)
        self._value = self._value_box(values, "value")
        self._join = QComboBox()
        self._join.addItems(JOINS)
        self._op2 = QComboBox()
        self._op2.addItems(OPS)
        self._value2 = self._value_box(values, "second value (optional)")
        grid.addWidget(self._op, 0, 0)
        grid.addWidget(self._value, 0, 1)
        grid.addWidget(self._join, 1, 0)
        grid.addWidget(QLabel("<span style='color: palette(mid)'>second condition (optional)</span>"), 1, 1)
        grid.addWidget(self._op2, 2, 0)
        grid.addWidget(self._value2, 2, 1)
        lay.addLayout(grid)
        current = model.filter_of(col)
        if current is not None:
            self._op.setCurrentText(current.conds[0][0])
            self._value.setText(current.conds[0][1])
            if len(current.conds) > 1:
                self._join.setCurrentText(current.join)
                self._op2.setCurrentText(current.conds[1][0])
                self._value2.setText(current.conds[1][1])
        buttons = QHBoxLayout()
        apply = QPushButton("Apply")
        apply.setDefault(True)
        apply.clicked.connect(self._apply)
        clear = QPushButton("Clear")
        clear.clicked.connect(self._clear)
        clear_all = QPushButton("Clear all")
        clear_all.clicked.connect(self._clear_all)
        clear_all.setEnabled(model.is_filtered())
        presets = QPushButton("Presets")
        presets.clicked.connect(self._presets)
        buttons.addWidget(apply)
        buttons.addWidget(clear)
        buttons.addWidget(clear_all)
        buttons.addWidget(presets)
        lay.addLayout(buttons)
        self._value.setFocus()

    def _value_box(self, values, placeholder):
        e = QLineEdit()
        e.setPlaceholderText(placeholder)
        e.setMinimumWidth(190)
        if values:
            comp = QCompleter(values, self)
            comp.setCaseSensitivity(Qt.CaseInsensitive)
            comp.setFilterMode(Qt.MatchContains)
            e.setCompleter(comp)
        e.returnPressed.connect(self._apply)
        return e

    def _apply(self):
        self._header.model().set_filter(self._col, self._op.currentText(), self._value.text(),
                                        self._join.currentText(), self._op2.currentText(), self._value2.text())
        self.close()

    def _clear(self):
        self._header.model().clear_filter(self._col)
        self.close()

    def _clear_all(self):
        self._header.model().clear_filters()
        self.close()

    def _presets(self):
        view = self._header.parent()
        pos = self.mapToGlobal(QPoint(0, self.height()))
        self.close()
        # after the popup is gone, the menu is a popup of its own
        QTimer.singleShot(0, lambda: preset_menu(view).exec(pos))


def _html(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class FilterHeader(QHeaderView):
    """horizontal header: click sorts, the arrow at the right end filters,
    a right click opens the presets menu"""

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

    def open_filter(self, logical):
        r = self.arrow_rect(logical)
        popup = FilterPopup(self, logical)
        popup.adjustSize()
        popup.move(self.viewport().mapToGlobal(QPoint(r.right() - popup.width(), r.bottom())))
        popup.show()
        return popup

    def mousePressEvent(self, e):
        logical = self._arrow_at(e.position().toPoint())
        if logical >= 0 and e.button() == Qt.LeftButton and isinstance(self.model(), FilterSortModel):
            self._arrow_press = True
            self.open_filter(logical)
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

    def contextMenuEvent(self, e):
        view = self.parent()
        if isinstance(view, FilterTableView) and isinstance(self.model(), FilterSortModel):
            preset_menu(view, self).exec(e.globalPos())
            e.accept()
            return
        super().contextMenuEvent(e)

    def viewportEvent(self, e):
        if e.type() == QEvent.ToolTip:
            pos = e.pos()
            logical = self.logicalIndexAt(pos)
            m = self.model()
            if logical >= 0 and isinstance(m, FilterSortModel):
                f = m.filter_of(logical)
                if self.arrow_rect(logical).contains(pos):
                    tip = ("Filter: " + f.describe()) if f else "Filter this column"
                else:
                    tip = "Click to sort (ascending, descending, original order)\nRight click: filter presets"
                    if f:
                        tip += "\nFiltered: " + f.describe()
                QToolTip.showText(e.globalPos(), tip, self)
                return True
        return super().viewportEvent(e)


class FilterTableView(QTableView):
    """QTableView with the sorting / filtering header. preset_kind groups
    the presets: tables of the same kind share them."""

    def __init__(self, model=None, parent=None, preset_kind="table"):
        super().__init__(parent)
        self.preset_kind = preset_kind
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

    # ---- presets -------------------------------------------------------
    def save_preset(self, name=None):
        m = self.model()
        if not m.is_filtered():
            return False
        if name is None:
            name, ok = QInputDialog.getText(self, "Save filter preset", "Name of the preset:",
                                            text=m.describe_filters()[:60])
            if not ok or not name.strip():
                return False
        presets = load_presets(self.preset_kind)
        presets[name.strip()] = m.filters_as_preset()
        save_presets(self.preset_kind, presets)
        return True

    def apply_preset(self, name, quiet=False):
        preset = load_presets(self.preset_kind).get(name)
        if preset is None:
            return None
        missing = self.model().apply_preset(preset)
        if missing and not quiet:
            QMessageBox.information(self, "Filter preset",
                                    "Not applied, this table has no column: " + ", ".join(missing))
        return missing

    def delete_preset(self, name):
        presets = load_presets(self.preset_kind)
        if presets.pop(name, None) is not None:
            save_presets(self.preset_kind, presets)


class ListTableModel(FilterSortModel):
    """a plain list of rows (tuples) with headings"""

    def __init__(self, headers, rows, right_aligned=()):
        super().__init__()
        self._headers = list(headers)
        self._data = [tuple(r) for r in rows]
        self._right = set(right_aligned)
        self.rebuild()

    def set_rows(self, rows):
        """new data, filters and sorting stay"""
        self._data = [tuple(r) for r in rows]
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
