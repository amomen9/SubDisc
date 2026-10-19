"""LabelRankingMatrixWindow with MatrixPlot: the base matrix, the subgroup
matrix and half their difference, green for positive, red for negative."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel

from ..java_compat import float_to_string
from ..label_ranking import LabelRanking
from . import common
from .pdf_window_2d import _UnitPlot


class MatrixPlot(_UnitPlot):
    def __init__(self, lrm, title):
        super().__init__()
        self._lrm = lrm
        self._title = title
        n = lrm.get_size()
        L = self._lines
        L += [((0, 0), (0, -1)), ((0, -1), (1, -1)), ((1, -1), (1, 0)), ((1, 0), (0, 0))]
        for i in range(n + 1):
            L.append(((i / float(n), 0.0), (i / float(n), 0.01)))
            L.append(((0.0, -i / float(n)), (-0.01, -i / float(n))))

    def paintEvent(self, _e):
        self._s = min(self.width(), self.height()) * 0.85
        g = QPainter(self)
        n = self._lrm.get_size()
        g.setPen(Qt.NoPen)
        for i in range(n):
            y = float(np.float32(i) / np.float32(n) - np.float32(1))
            for j in range(n):
                x = j / float(n)
                v = int(np.float32(255) * np.float32(self._lrm.get(i, j)))
                neg = v < 0
                v = max(min(abs(v), 255), 0)
                if neg:
                    g.setBrush(QColor(255, 255 - v, 255 - v))   # red
                else:
                    g.setBrush(QColor(255 - v, 255, 255 - v))   # green
                g.drawRect(self._rect(x, y, 1.0 / n, 1.0 / n))
        g.setPen(QPen(Qt.black, 1.0))
        for (u1, v1), (u2, v2) in self._lines:
            g.drawLine(self._p(u1, v1), self._p(u2, v2))
        f = QFont(self.font())
        f.setPixelSize(11)
        g.setFont(f)
        for i in range(n):
            g.drawText(self._p((i + 0.5) / n, 0.04), LabelRanking.get_letter(i))
            g.drawText(self._p(-0.07, -(i + 0.5) / n), LabelRanking.get_letter(n - i - 1))
        g.drawText(self._p(0.4, -1.04), self._title)
        g.end()


class LabelRankingMatrixWindow(common.Window):
    def __init__(self, base_lrm, lrm, title):
        super().__init__()
        row = QHBoxLayout()
        row.addWidget(MatrixPlot(base_lrm, "Base Matrix"), 1)
        if lrm is not None:
            row.addWidget(MatrixPlot(lrm, "Subgroup Matrix"), 1)
            diff = base_lrm.clone()
            diff.subtract(lrm)
            diff.divide(2.0)
            diff.print()
            row.addWidget(MatrixPlot(diff, "Difference"), 1)
            self.content().addLayout(row, 1)
            html = ("<table cellspacing='4'><tr><th>&nbsp;</th><th>Pairwise Max</th></tr>"
                    "<tr><td>Dataset</td><td>%s (%s) </td></tr>"
                    "<tr><td>Subgroup</td><td>%s (%s) </td></tr>"
                    "<tr><td>Difference</td><td> %s (%s) </td></tr></table><br>"
                    "Most different label: %s") % (
                base_lrm.find_max(), float_to_string(base_lrm.pairwise_max(base_lrm)),
                lrm.find_max(), float_to_string(lrm.pairwise_max(lrm)),
                diff.find_max(), float_to_string(diff.pairwise_max(diff)),
                lrm.find_max_label())
            lbl = QLabel(html)
            lbl.setTextFormat(Qt.RichText)
            lbl.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
            self.content().addWidget(lbl, 1)
        else:
            row.addStretch(2)
            self.content().addLayout(row, 1)
        self.add_button_row([common.build_button("Close", "C", self.close)])
        self.setWindowTitle(title + " : Label Ranking Matrix")
        self.show_at(100, 100, common.MATRIX_WINDOW_DEFAULT_SIZE)
