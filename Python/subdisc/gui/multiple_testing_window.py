"""Multiple Testing: corrected p-values for the subgroups of a result
window (see multiple_testing.py). Not in java."""

import math

import numpy as np
from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QMessageBox

from .. import multiple_testing as mt
from . import common
from .feature_common import TableResultWindow

SOURCE_EXACT = "exact test from the counts"
SOURCE_COLUMN = "p-Value column (Gaussian / empirical)"


def open_multiple_testing(parent, table, selection, subgroups, search_parameters, sd):
    tt = search_parameters.get_target_type()
    sources = []
    if mt.has_exact_test(tt):
        sources.append(SOURCE_EXACT)
    column_ok = sum(1 for s in subgroups if 0.0 <= s.get_p_value() <= 1.0)
    if column_ok:
        sources.append(SOURCE_COLUMN)
    if not sources:
        QMessageBox.information(parent, "Multiple Testing",
                                "There are no p-values yet. For this target type compute them first with "
                                "'Gaussian p-Values' or 'Empirical p-Values'.")
        return None
    return MultipleTestingWindow(table, selection, subgroups, search_parameters, sd, sources)


class MultipleTestingWindow(TableResultWindow):
    HEADERS = ("Nr.", "Depth", "Conditions", "Quality", "p-Value", "Adjusted p / Threshold", "Significant")

    def __init__(self, table, selection, subgroups, search_parameters, sd, sources):
        self._subgroups = list(subgroups)
        self._sp = search_parameters
        self._counts = sd.get_candidate_count_per_depth() if sd is not None else {}
        self._candidates = sd.get_candidate_count() if sd is not None else 0
        self._exact = None
        if SOURCE_EXACT in sources:
            common.set_busy(True)
            try:
                self._exact = mt.exact_p_values(self._subgroups, search_parameters, table, selection)
            finally:
                common.set_busy(False)
        super().__init__("Multiple Testing - %d subgroups" % len(self._subgroups), self.HEADERS, [], "",
                         "subgroups", preset_kind="multiple_testing")
        bar = QHBoxLayout()
        self._source = QComboBox()
        self._source.addItems(sources)
        self._method = QComboBox()
        self._method.addItems(mt.METHODS)
        self._method.setCurrentIndex(2)
        self._m = QComboBox()
        m_items = ["subgroups shown (%d)" % len(self._subgroups)]
        if self._candidates > len(self._subgroups):
            m_items.insert(0, "candidates evaluated (%s)" % "{:,}".format(self._candidates))
        self._m.addItems(m_items)
        self._alpha = QDoubleSpinBox()
        self._alpha.setDecimals(3)
        self._alpha.setRange(0.001, 0.5)
        self._alpha.setSingleStep(0.01)
        self._alpha.setValue(0.05)
        for lbl, w in (("p-values", self._source), ("method", self._method), ("m", self._m), ("alpha", self._alpha)):
            bar.addWidget(QLabel(lbl))
            bar.addWidget(w)
        bar.addStretch(1)
        self.content().insertLayout(1, bar)
        for w in (self._source, self._method, self._m):
            w.currentIndexChanged.connect(lambda _i: self._recompute())
        self._alpha.valueChanged.connect(lambda _v: self._recompute())
        self._recompute()

    def _p_values(self):
        if self._source.currentText() == SOURCE_EXACT:
            return np.array(self._exact, dtype=np.float64)
        out = []
        for s in self._subgroups:
            p = s.get_p_value()
            out.append(p if 0.0 <= p <= 1.0 else float('nan'))
        return np.array(out, dtype=np.float64)

    def _recompute(self):
        p = self._p_values()
        ok = ~np.isnan(p)
        method = self._method.currentText()
        alpha = self._alpha.value()
        m_text = self._m.currentText()
        m = self._candidates if m_text.startswith("candidates") else len(self._subgroups)
        m = max(m, int(np.count_nonzero(ok)))
        depths = [s.get_depth() for s in self._subgroups]
        adj = np.full(p.shape[0], np.nan)
        sig = np.zeros(p.shape[0], dtype=bool)
        if ok.any():
            counts = self._counts
            if method == mt.METHODS[4] and not counts:
                counts = {}
                for d in depths:
                    counts[d] = counts.get(d, 0) + 1
            a, s, thr = mt.adjust(method, p[ok], m, alpha, [d for d, k in zip(depths, ok) if k], counts,
                                  self._sp.get_search_depth())
            adj[ok] = a if a is not None else thr
            sig[ok] = s
        rows = []
        for i, s in enumerate(self._subgroups):
            rows.append((s.get_id(), s.get_depth(), str(s.get_conditions()), float(s.get_measure_value()),
                         float(p[i]), float(adj[i]), "yes" if sig[i] else "no"))
        self.set_rows(rows)
        layered = method == mt.METHODS[4]
        alt = mt.alternative_for(self._sp.get_quality_measure()) if self._source.currentText() == SOURCE_EXACT else None
        text = ("<b>%d of %d</b> subgroups significant at alpha = %s with %s%s." %
                (int(np.count_nonzero(sig)), len(self._subgroups), _g(alpha), method,
                 "" if layered else ", m = %s" % "{:,}".format(m)))
        if layered:
            text += " Threshold per depth L: alpha / (D * C_L), D = %d, C_L = %s." % (
                self._sp.get_search_depth(), ", ".join("%d: %s" % (k, "{:,}".format(v)) for k, v in sorted(self._counts.items())) or "-")
        if alt:
            text += " Exact test: %s." % alt
        if (~ok).any():
            text += " %d subgroups have no p-value." % int(np.count_nonzero(~ok))
        text += ("<br>m = subgroups shown is optimistic: the search tested every candidate. "
                 "BH and BY control the share of false discoveries, Bonferroni and Holm the chance of any.")
        self.set_summary(text)


def _g(x):
    return ("%g" % x) if not math.isnan(x) else "-"
