"""ModelWindow: the 'Base Model' and 'Show Model' pictures.

- SINGLE_NUMERIC / SCAPE: density of the numeric target (data and subgroup,
  or positives and negatives)
- DOUBLE_REGRESSION / DOUBLE_CORRELATION: scatter plot (with regression line)
- MULTI_LABEL: the Bayesian network
"""

import numpy as np
from PySide6.QtWidgets import QCheckBox

from .. import log
from ..java_compat import format_fixed
from . import common
from .charts import ChartPanel
from .dag_view import DAGView

BASE_MODEL_TEXT = "Base Model (all data)"
SAMPLE_SIZE = 200_000
_FIRST_PAINT = (1.0, 0x55 / 255.0, 0x55 / 255.0)


class ModelWindow(common.Window):
    def __init__(self):
        super().__init__()
        self._relative = None
        self._sample = None

    # ------------------------------------------------------------------
    # SINGLE_NUMERIC and SCAPE
    # ------------------------------------------------------------------

    @classmethod
    def numeric(cls, domain, dataset_pdf, subgroup_pdf, title, is_scape):
        w = cls()
        add_subgroup = subgroup_pdf is not None
        w._dataset_pdf = dataset_pdf
        w._subgroup_pdf = subgroup_pdf
        w._is_scape = is_scape
        w._chart = ChartPanel()
        w.content().addWidget(w._chart, 1)
        if is_scape:
            w._names = ("positives", "negatives")
        else:
            w._names = ("data", "subgroup")   # XXX MM (java) sometimes complement, not dataset
        w._draw_pdf(True)
        ax = w._chart.ax
        ax.set_xlabel(domain.get_name())
        ax.set_ylabel("density")
        buttons = []
        if add_subgroup:
            w._relative = QCheckBox("relative")
            w._relative.setChecked(True)
            w._relative.toggled.connect(lambda on: w._draw_pdf(on))
            buttons.append(w._relative)
        buttons.append(common.build_button("Close", "C", w.close))
        w.add_button_row(buttons)
        w.setWindowTitle(title + ": Numeric Distribution")
        w.show_at(50, 50, common.WINDOW_DEFAULT_SIZE)
        return w

    def _draw_pdf(self, relative):
        d, s = self._dataset_pdf, self._subgroup_pdf
        n = d.size()
        xs = [d.get_middle(i) for i in range(n)]
        ys = [d.get_density(i) for i in range(n)]
        ax = self._chart.ax
        for ln in list(ax.lines):
            ln.remove()
        if s is not None:
            scale = s.get_absolute_count() / float(d.get_absolute_count()) if relative else 1.0
            sx = [s.get_middle(i) for i in range(n)]
            sy = [s.get_density(i) * scale for i in range(n)]
            if self._is_scape:
                d_paint, s_paint, d_width, s_width = "red", "blue", 2.5, 2.5
            else:
                d_paint, s_paint, d_width, s_width = "black", "lightgray", 1.5, 2.5
            # series 0 is the subgroup, series 1 the data, painted with
            # swapped paints like in java
            ax.plot(xs, ys, color=s_paint, linewidth=s_width, label=self._names[0])
            ax.plot(sx, sy, color=d_paint, linewidth=d_width, label=self._names[1])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        else:
            ax.plot(xs, ys, color="black", linewidth=2.5)
        ax.relim()
        ax.autoscale_view()
        self._chart.redraw()

    # ------------------------------------------------------------------
    # DOUBLE_CORRELATION and DOUBLE_REGRESSION
    # ------------------------------------------------------------------

    @classmethod
    def scatter(cls, table, x_column, y_column, rm, subgroup):
        w = cls()
        w._table = table
        w._x = x_column
        w._y = y_column
        w._rm = rm
        w._subgroup = subgroup
        if table.get_nr_rows() > SAMPLE_SIZE:
            w._sample = table.get_random_bit_set(SAMPLE_SIZE)
        is_regression = rm is not None
        for_subgroup = subgroup is not None
        if is_regression:
            name = "%s = %s + %s * %s" % (y_column.get_name(), _ffix(rm.get_intercept()),
                                          _ffix(rm.get_slope()), x_column.get_name())
        elif for_subgroup:
            name = "2D distribution (r = %s)" % _ffix(subgroup.get_secondary_statistic())
        else:
            name = "2D distribution"
        if for_subgroup:
            name += "\n(%s, %d examples)" % (subgroup, subgroup.get_coverage())
        else:
            name += " (all data)"

        w._chart = ChartPanel()
        ax = w._chart.ax
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel(x_column.get_name())
        ax.set_ylabel(y_column.get_name())
        w._points = []
        w._draw_points()
        if is_regression:
            lo, hi = x_column.get_min(), x_column.get_max()
            ax.plot([lo, hi], [rm.get_base_function_value(lo), rm.get_base_function_value(hi)],
                    color=_FIRST_PAINT, linewidth=2.0)
        w.content().addWidget(w._chart, 1)
        buttons = []
        if w._sample is not None:
            buttons.append(common.build_button("Resample", "R", w._resample))
        buttons.append(common.build_button("Close", "C", w.close))
        w.add_button_row(buttons)
        who = _create_title(subgroup) if for_subgroup else BASE_MODEL_TEXT
        kind = "Regression" if is_regression else "Correlation"
        w.setWindowTitle("%s: %s%s" % (who, kind, "" if w._sample is None else " (sampled)"))
        w.show_at(50, 50, common.WINDOW_DEFAULT_SIZE)
        return w

    def _data_points(self):
        n = self._x.size()
        xs = self._x.floats_view()[:n]
        ys = self._y.floats_view()[:n]
        sub = None
        if self._subgroup is not None:
            m = self._subgroup.get_members()[:n]
            if self._sample is not None:
                m = m & self._sample[:n]
            sub = (xs[m], ys[m])
        if self._sample is None:
            data = (xs, ys)
        else:
            log.log_command_line("Sampling before plotting dataset.")
            data = (xs[self._sample[:n]], ys[self._sample[:n]])
        return sub, data

    def _draw_points(self):
        ax = self._chart.ax
        for p in self._points:
            p.remove()
        self._points = []
        sub, data = self._data_points()
        # the data first, so the subgroup is drawn on top of it
        color = "lightgray" if sub is not None else "black"
        self._points += ax.plot(data[0], data[1], linestyle="none", marker="s", markersize=1.6,
                                markeredgewidth=0, color=color)
        if sub is not None:
            self._points += ax.plot(sub[0], sub[1], linestyle="none", marker="s", markersize=1.6,
                                    markeredgewidth=0, color="black")
        self._chart.redraw()

    def _resample(self):
        log.log_command_line("attempting to resample")
        self._draw_points()

    # ------------------------------------------------------------------
    # MULTI_LABEL
    # ------------------------------------------------------------------

    @classmethod
    def bayesian_network(cls, dag, width, height):
        w = cls()
        view = DAGView(dag)
        view.set_dag_area(width, height)
        view.draw_dag()
        w.content().addWidget(view, 1)
        w.add_button_row([common.build_button("Close", "C", w.close)])
        w.setWindowTitle(BASE_MODEL_TEXT + ": Bayesian Network")
        w.show_at(0, 0)
        w.resize(width, height)
        return w


def _ffix(v):
    """String.format("%f", (float) v), java formats the float as a double"""
    return format_fixed(float(np.float32(v)), 6)


def _create_title(subgroup):
    return "Subgroup %d (%s)" % (subgroup.get_id(), subgroup)
