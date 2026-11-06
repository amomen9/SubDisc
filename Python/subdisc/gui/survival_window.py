"""Kaplan-Meier curves for the survival target type (Show Model and Base
Model). Not in java."""

import math

import numpy as np

from ..survival import SurvivalBase, chi2_p_value, km_median
from . import common
from .charts import ChartPanel


class KaplanMeierWindow(common.Window):
    """the subgroup, its complement and the whole data; base model when
    members is None"""

    def __init__(self, time_column, event_column, selection, members=None, title="Base Model"):
        super().__init__(title)
        base = SurvivalBase(time_column, event_column, selection)
        chart = ChartPanel(self)
        ax = chart.ax
        info = []
        if members is None:
            self._curve(ax, base, None, "all rows", "#4c72b0", info)
        else:
            m = np.asarray(members, dtype=bool)
            self._curve(ax, base, m, "subgroup", "#c44e52", info)
            comp = np.zeros(base.nr_rows, dtype=bool)
            comp[base.rows] = True
            comp[:m.shape[0]] &= ~m[:comp.shape[0]]
            self._curve(ax, base, comp, "complement", "#4c72b0", info)
            self._curve(ax, base, None, "all rows", "#999999", info, dashed=True)
            z, chi2, o, e, hr = base.log_rank(m)
            info.append("log-rank: chi2 = %.3f, p = %.3g, z = %.3f; events %d observed, %.1f expected; hazard ratio %.3f"
                        % (chi2, chi2_p_value(chi2), z, o, e, hr))
        ax.set_xlabel(time_column.get_name())
        ax.set_ylabel("survival probability S(t)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(left=0)
        ax.legend(loc="best", fontsize=9)
        ax.set_title(title if len(title) < 90 else title[:87] + "...", fontsize=10)
        self.content().addWidget(chart, 1)
        from PySide6.QtWidgets import QLabel
        lbl = QLabel("\n".join(info))
        lbl.setWordWrap(True)
        self.content().addWidget(lbl)
        self.add_button_row([common.build_button("Close", "C", self.close, common.BUTTON_MEDIUM_SIZE)])
        chart.redraw()
        self.show_at(220, 140, common.WINDOW_DEFAULT_SIZE)

    @staticmethod
    def _curve(ax, base, members, label, color, info, dashed=False):
        xs, ys, censored = base.kaplan_meier(members)
        n = int(base.size if members is None else np.count_nonzero(base.sorted_mask(members)))
        med = km_median(xs, ys)
        # steps: the curve drops at every event time and stays flat after the last one
        last = float(base.times.max()) if base.size else 0.0
        x = np.concatenate((xs, [max(last, xs[-1])]))
        y = np.concatenate((ys, [ys[-1]]))
        ax.step(x, y, where="post", color=color, linewidth=1.6, linestyle="--" if dashed else "-",
                label="%s (n = %d, median %s)" % (label, n, "not reached" if math.isnan(med) else "%.3g" % med))
        if len(censored) and not dashed:
            # a small tick where a row left the study without the event
            idx = np.searchsorted(xs, censored, side="right") - 1
            ax.plot(censored, ys[np.maximum(idx, 0)], linestyle="none", marker="|", markersize=6, color=color, alpha=0.6)
        info.append("%s: %d rows, median %s" % (label, n, "not reached" if math.isnan(med) else "%.4g" % med))
