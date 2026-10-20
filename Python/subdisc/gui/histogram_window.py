"""Explore: stacked bar chart of an attribute against a target, plus the
CrossTable and the (numeric) PlotWindow."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QHBoxLayout, QLabel, QRadioButton, QScrollArea,
                               QSlider, QVBoxLayout,
                               QAbstractItemView)
from matplotlib.ticker import MaxNLocator

from .. import log
from ..attribute_type import AttributeType
from ..java_compat import float_to_string
from . import common
from .charts import ChartPanel, style_axes
from .table_filter import FilterTableView, ListTableModel

# no legend when the target has many values
TARGET_VALUES_MAX = 30
ATTRIBUTE_VALUES_MAX = 100
MIN_NR_BINS = 1   # all data
DEFAULT_NR_BINS = 8
MAX_NR_BINS = 32

# JFreeChart DefaultDrawingSupplier paint sequence, so the colours look the same
_PAINTS = [
    (0xFF, 0x55, 0x55), (0x55, 0x55, 0xFF), (0x55, 0xFF, 0x55), (0xFF, 0xFF, 0x55),
    (0xFF, 0x55, 0xFF), (0x55, 0xFF, 0xFF), (255, 175, 175), (128, 128, 128),
    (0xC0, 0, 0), (0, 0, 0xC0), (0, 0xC0, 0), (0xC0, 0xC0, 0), (0xC0, 0, 0xC0),
    (0, 0xC0, 0xC0), (64, 64, 64), (0xFF, 0x40, 0x40), (0x40, 0x40, 0xFF),
    (0x40, 0xFF, 0x40), (0xFF, 0xFF, 0x40), (0xFF, 0x40, 0xFF), (0x40, 0xFF, 0xFF),
    (192, 192, 192), (0x80, 0, 0), (0, 0, 0x80), (0, 0x80, 0), (0x80, 0x80, 0),
    (0x80, 0, 0x80), (0, 0x80, 0x80), (0xFF, 0x80, 0x80), (0x80, 0x80, 0xFF),
    (0x80, 0xFF, 0x80), (0xFF, 0xFF, 0x80), (0xFF, 0x80, 0xFF), (0x80, 0xFF, 0xFF),
]
PAINTS = [(r / 255.0, g / 255.0, b / 255.0) for r, g, b in _PAINTS]


def _key_text(k):
    if isinstance(k, bool):
        return "true" if k else "false"
    if isinstance(k, float):
        return float_to_string(k)
    return str(k)


class HistogramWindow(common.Window):
    def __init__(self, table):
        super().__init__("Histogram")
        if table is None or table.get_nr_columns() == 0:
            return
        self._table = table
        names = [table.get_column(i).get_name() for i in range(table.get_nr_columns())]

        self._chart = ChartPanel()
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._chart)
        self.content().addWidget(self._scroll, 1)

        south = QHBoxLayout()
        # attribute
        ap = common.build_border("Select Attribute")
        al = QVBoxLayout(ap)
        self._a_box = common.build_combo_box(names, lambda: self._update(True))
        al.addWidget(self._a_box)
        self._a_slider = self._slider(lambda: self._update(True))
        al.addLayout(self._slider_box(self._a_slider))
        self._a_plot = common.build_button("Plot Attribute", "A", lambda: PlotWindow(table.get_column(self._a_box.currentIndex())))
        al.addWidget(self._a_plot)
        south.addWidget(ap)
        # target (duplicate code)
        tp = common.build_border("Select Target")
        tl = QVBoxLayout(tp)
        self._t_box = common.build_combo_box(names, lambda: self._update(False))
        tl.addWidget(self._t_box)
        self._t_slider = self._slider(lambda: self._update(False))
        tl.addLayout(self._slider_box(self._t_slider))
        self._t_plot = common.build_button("Plot Target", "T", lambda: PlotWindow(table.get_column(self._t_box.currentIndex())))
        tl.addWidget(self._t_plot)
        south.addWidget(tp)
        # misc
        mp = common.build_border("Other")
        ml = QVBoxLayout(mp)
        ml.addWidget(common.build_button("CrossTable", "R", self._cross_table))
        close = common.build_button("Close", "C", self.close)
        ml.addWidget(close)
        ml.addStretch(1)
        south.addWidget(mp)
        self.content().addLayout(south)

        self._a_map = {}
        self._t_map = {}
        self._dataset = None
        self._update_map(True)
        self._update_map(False)
        self._update_chart()
        close.setFocus()
        self.show_at(50, 50, common.WINDOW_DEFAULT_SIZE)

    @staticmethod
    def _slider(slot):
        s = QSlider(Qt.Horizontal)
        s.setRange(MIN_NR_BINS, MAX_NR_BINS)
        s.setValue(DEFAULT_NR_BINS)
        s.setTickPosition(QSlider.TicksBelow)
        s.setTickInterval(1)
        s.setPageStep(1)
        # only when the user lets go, like getValueIsAdjusting()
        s.valueChanged.connect(lambda _v: None if s.isSliderDown() else slot())
        s.sliderReleased.connect(slot)
        return s

    @staticmethod
    def _slider_box(slider):
        v = QVBoxLayout()
        v.addWidget(slider)
        labels = QHBoxLayout()
        for i, t in enumerate(("1", "8", "16", "24", "32")):
            if i:
                labels.addStretch(1)
            labels.addWidget(QLabel(t))
        v.addLayout(labels)
        return v

    def _update(self, is_attribute):
        self._update_map(is_attribute)
        self._update_chart()

    # NOTE (java) BINARY uses String for now, 4 use cases instead of 9
    def _update_map(self, is_attribute):
        box = self._a_box if is_attribute else self._t_box
        c = self._table.get_column(box.currentIndex())
        t = c.get_type()
        slider = self._a_slider if is_attribute else self._t_slider
        plot = self._a_plot if is_attribute else self._t_plot
        slider.setEnabled(t == AttributeType.NUMERIC)
        plot.setEnabled(t == AttributeType.NUMERIC)
        m = {}
        if t == AttributeType.NOMINAL:
            for i, v in enumerate(c.get_domain()):
                m[v] = i
        elif t == AttributeType.NUMERIC:
            self._safarii_histo(c, m, slider.value())
        elif t == AttributeType.BINARY:
            # the order [0, 1] is essential
            m[False] = 0
            m[True] = 1
        else:
            self._unknown("updateMap", t)
            return
        if is_attribute:
            self._a_map = m
        else:
            self._t_map = m

    @staticmethod
    def _safarii_histo(column, m, nr_bins):
        """NOTE (java) not safe for overflow / NaN"""
        F = np.float32
        n = column.size()
        vals = column.floats_view()[:n]
        ok = vals[~column.missing_view()[:n]]
        count = int(ok.shape[0])
        s = np.cumsum(ok, dtype=np.float32)[-1] if count else F(0.0)
        avg = F(s / F(count)) if count > 0 else F(0.0)
        sd = F(0.0)
        for x in ok.tolist():
            d = float(F(avg - F(x)))
            sd = F(float(sd) + d * d)
        sd = F(np.sqrt(float(sd)))
        start = max(F(avg - F(F(2.3) * sd)), F(column.get_min()))
        stop = min(F(avg + F(F(2.3) * sd)), F(column.get_max()))
        if start == stop:
            start = F(start - F(1))
            stop = F(stop + F(1))
        step = F(F(stop - start) / F(nr_bins - 2)) if nr_bins > 2 else F(0.0)
        for i in range(nr_bins - 1):
            m.setdefault(float(F(start + F(F(i) * step))), i)
        m[float("inf")] = nr_bins - 1

    def _bins(self, column, keys):
        n = column.size()
        k = np.asarray(keys, dtype=np.float32)
        idx = np.searchsorted(k, column.floats_view()[:n], side="left")
        # java would go out of bounds on NaN, keep it in the last bin
        return np.minimum(idx, len(k) - 1)

    def _create_dataset(self):
        a = self._table.get_column(self._a_box.currentIndex())
        t = self._table.get_column(self._t_box.currentIndex())
        am, tm = self._a_map, self._t_map
        counts = np.zeros((len(am), len(tm)), dtype=np.int64)
        n = a.size()
        at, tt = a.get_type(), t.get_type()

        def idx(col, typ, m):
            if typ == AttributeType.NOMINAL:
                codes = col.nominal_codes_view()[:n]
                dv = col.distinct_values()
                lut = np.array([m.get(v, -1) for v in dv], dtype=np.int64)
                return lut[codes]
            if typ == AttributeType.NUMERIC:
                return self._bins(col, list(m.keys()))
            if typ == AttributeType.BINARY:
                return col.binaries_view()[:n].astype(np.int64)
            return None

        ai = idx(a, at, am)
        ti = idx(t, tt, tm)
        if ai is None or ti is None:
            self._unknown("createDataset", at if ai is None else tt)
            return am, tm, counts
        keep = np.ones(n, dtype=bool)
        # numeric parts skip missing values (both columns checked)
        if at == AttributeType.NUMERIC and tt != AttributeType.BINARY:
            keep &= ~a.missing_view()[:n] & ~t.missing_view()[:n]
        elif tt == AttributeType.NUMERIC and at == AttributeType.NOMINAL:
            keep &= ~a.missing_view()[:n] & ~t.missing_view()[:n]
        np.add.at(counts, (ai[keep], ti[keep]), 1)
        return am, tm, counts

    def _update_chart(self):
        am, tm, counts = self._create_dataset()
        self._dataset = (list(am.keys()), list(tm.keys()), counts)
        show_legend = len(tm) < TARGET_VALUES_MAX
        fig = self._chart.figure
        fig.clear()
        ax = fig.add_subplot(111)
        style_axes(ax)
        ax.grid(False, axis="x")
        x = np.arange(len(am))
        bottom = np.zeros(len(am))
        a_labels = [_key_text(k) for k in am.keys()]
        for j, tk in enumerate(tm.keys()):
            col = counts[:, j] if counts.size else np.zeros(len(am))
            ax.bar(x, col, bottom=bottom, width=0.8, color=PAINTS[j % len(PAINTS)], label=_key_text(tk), linewidth=0)
            bottom += col
        ax.set_xticks(x)
        many = len(am) > ATTRIBUTE_VALUES_MAX
        if many:
            ax.set_xticklabels([])
            ax.grid(False)
            ax.tick_params(axis="y", labelsize=6)
        else:
            ax.set_xticklabels(a_labels, rotation=0 if len(am) < 12 else 90, fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if len(am):
            ax.set_xlim(-0.6, len(am) - 0.4)
        if show_legend and len(tm):
            leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08 if not many else -0.02),
                            ncol=min(len(tm), 8), frameon=many, fontsize=6 if many else 8)
            if many:
                leg.get_frame().set_edgecolor("black")
        self._chart.setMinimumWidth(max(400, len(am) * 20))
        self._chart.redraw()

    def _cross_table(self):
        if self._dataset is not None:
            CrossTableWindow(*self._dataset)

    def _unknown(self, source, t):
        log.log_command_line("HistogramWindow.%s(), unknown AttributeType: '%s'" % (source, t))


class CrossTableWindow(common.Window):
    """Attributes are the rows, targets the columns (no header)."""

    def __init__(self, attribute_keys, target_keys, counts):
        super().__init__("CrossTable")
        # java puts the target values in the first row of a header-less
        # table; here they are real headings, so the columns can be sorted
        # and filtered
        headers = ["Attribute v / Target >"] + [_key_text(k) for k in target_keys]
        rows = [[_key_text(a)] + [int(counts[i, j]) for j in range(len(target_keys))]
                for i, a in enumerate(attribute_keys)]
        t = FilterTableView(ListTableModel(headers, rows), preset_kind="cross")
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.verticalHeader().setVisible(False)
        t.resizeColumnsToContents()
        t.fit_headings()   # room for the sort and filter arrows
        self.content().addWidget(t, 1)
        self.add_button_row([common.build_button("Close", None, self.close)])
        w = min(1200, sum(t.columnWidth(c) for c in range(len(headers))) + 40)
        h = min(800, 22 * (len(rows) + 1) + 120)
        self.show_at(50, 50)
        self.resize(max(w, 300), max(h, 160))


class PlotWindow(common.Window):
    """values of a numeric column in the original and in sorted order"""

    def __init__(self, column):
        super().__init__()
        if column is None:
            return
        self._chart = ChartPanel()
        n = column.size()
        o = column.floats_view()[:n].astype(np.float64)
        s = np.sort(o)   # NaN goes last, like Arrays.sort()
        ax = self._chart.ax
        x = np.arange(n)
        (self._orig,) = ax.plot(x, o, color="black", linewidth=1.0)
        (self._sorted,) = ax.plot(x, s, color="red", linewidth=1.0)
        self._sorted.set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.content().addWidget(self._chart, 1)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(QLabel("Plot Data for: "))
        grp = QButtonGroup(self)
        for i, (text, key) in enumerate((("Original", "original"), ("Sorted", "sorted"), ("Both", "both"))):
            rb = QRadioButton(text)
            rb.toggled.connect(lambda on, k=key: on and self._show(k))
            grp.addButton(rb)
            row.addWidget(rb)
            if i == 0:
                rb.setChecked(True)
        row.addWidget(common.build_button("Close", "C", self.close))
        row.addStretch(1)
        self.content().addLayout(row)
        self.setWindowTitle("PlotWindow " + column.get_name())
        self.show_at(50, 50, common.WINDOW_DEFAULT_SIZE)

    def _show(self, key):
        self._orig.set_visible(key != "sorted")
        self._sorted.set_visible(key != "original")
        self._chart.redraw()
