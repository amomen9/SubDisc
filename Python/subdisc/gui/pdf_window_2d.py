"""Two dimensional density pictures for MULTI_NUMERIC (java PDFWindow2D,
PDFPlot and PDFPlot2). All drawing is done in a unit square, the same way
the java panels do it: x in [0, 1] to the right, y in [-1, 0] upwards."""

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QHBoxLayout, QWidget

from . import common


class _UnitPlot(QWidget):
    """maps unit coordinates (u, v) to pixels: ((u + 0.15) * s, (v + 1.1) * s)"""

    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)
        self.setMinimumSize(120, 120)
        self._s = 1.0
        self._lines = []   # [((u1, v1), (u2, v2))]

    def _p(self, u, v):
        return QPointF((u + 0.15) * self._s, (v + 1.1) * self._s)

    def _rect(self, u, v, w, h):
        return QRectF(self._p(u, v), self._p(u + w, v + h))

    def _box_and_ticks(self, nx, ny):
        L = self._lines
        L += [((0, 0), (0, -1)), ((0, -1), (1, -1)), ((1, -1), (1, 0)), ((1, 0), (0, 0))]
        for i in range(nx + 1):
            L.append(((i / float(nx), 0.0), (i / float(nx), 0.01)))
        for i in range(ny + 1):
            L.append(((0.0, -i / float(ny)), (-0.01, -i / float(ny))))

    def _bounding_box(self, gx0, gx1, gy0, gy1, dx0, dx1, dy0, dy1):
        sx = gx1 - gx0
        sy = gy1 - gy0
        a = (dx0 - gx0) / sx
        b = (dx1 - gx0) / sx
        c = (gy0 - dy0) / sy
        d = (gy0 - dy1) / sy
        self._lines += [((a, c), (b, c)), ((a, d), (b, d)), ((a, c), (a, d)), ((b, c), (b, d))]

    def _draw_lines_and_text(self, g, title, x_axis, y_axis):
        g.setPen(QPen(Qt.black, 1.0))
        for (u1, v1), (u2, v2) in self._lines:
            g.drawLine(self._p(u1, v1), self._p(u2, v2))
        f = QFont(self.font())
        f.setPixelSize(11)
        g.setFont(f)
        g.drawText(self._p(0.4, 0.06), x_axis)
        g.drawText(self._p(0.01, -0.52), y_axis)
        g.drawText(self._p(0.5, -1.02), title)


class PDFPlot(_UnitPlot):
    """base model, grey scale, from a ProbabilityDensityFunction2D"""

    def __init__(self, pdf, title, x_axis, y_axis):
        super().__init__()
        self._pdf = pdf
        self._title, self._x_axis, self._y_axis = title, x_axis, y_axis
        self._box_and_ticks(pdf.get_size_x(), pdf.get_size_y())
        self._bounding_box(pdf.get_low_x(), pdf.get_high_x(), pdf.get_low_y(), pdf.get_high_y(),
                           pdf.get_min_x(), pdf.get_max_x(), pdf.get_min_y(), pdf.get_max_y())

    def paintEvent(self, _e):
        self._s = min(self.width(), self.height()) * 0.85
        g = QPainter(self)
        pdf = self._pdf
        nx, ny = pdf.get_size_x(), pdf.get_size_y()
        mx = pdf.get_max_density()
        dens = pdf.density
        g.setPen(Qt.NoPen)
        for i in range(ny):
            y = np.float32(ny - i) / np.float32(ny) - 1
            for j in range(nx):
                x = j / float(nx)
                v = int(255 * (dens[j, i] / mx)) if mx else 0
                v = max(min(v, 255), 0)
                g.setBrush(QColor(255 - v, 255 - v, 255 - v))
                g.drawRect(self._rect(x, float(y), 1.0 / nx, 1.0 / ny))
        self._draw_lines_and_text(g, self._title, self._x_axis, self._y_axis)
        g.end()


class PDFPlot2(_UnitPlot):
    """subgroup / complement / difference, values in [-1, 1], green and red"""

    def __init__(self, grid, stats, limits, title, x_axis, y_axis):
        super().__init__()
        self._grid = np.asarray(grid, dtype=np.float64)
        self._title, self._x_axis, self._y_axis = title, x_axis, y_axis
        nx, ny = self._grid.shape
        self._box_and_ticks(nx, ny)
        # stats = [x_min, x_max, x_n, y_min, y_max, y_n, dx, dy]
        self._bounding_box(stats[0], stats[1], stats[3], stats[4],
                           limits[0][0], limits[0][1], limits[1][0], limits[1][1])

    def paintEvent(self, _e):
        self._s = min(self.width(), self.height()) * 0.85
        g = QPainter(self)
        nx, ny = self._grid.shape
        g.setPen(Qt.NoPen)
        for i in range(nx):
            x = i / float(nx)
            for j in range(ny - 1, -1, -1):
                y = (j + 1.0) / ny
                v = int(np.rint(255.0 * self._grid[i, j]))
                v = max(min(v, 255), -255)
                if v < 0:
                    g.setBrush(QColor(255, v + 255, v + 255))   # red
                else:
                    g.setBrush(QColor(255 - v, 255, 255 - v))   # green
                g.drawRect(self._rect(x, -y, 1.0 / nx, 1.0 / ny))
        self._draw_lines_and_text(g, self._title, self._x_axis, self._y_axis)
        g.end()


class PDFWindow2D(common.Window):
    @classmethod
    def base(cls, pdf, title, x_axis, y_axis):
        # TODO MM (java) make Base Model use PDFPlot2
        w = cls()
        row = QHBoxLayout()
        row.addWidget(PDFPlot(pdf, "Dataset PDF", x_axis, y_axis))
        w.content().addLayout(row, 1)
        w.add_button_row([common.build_button("Close", "C", w.close)])
        w.setWindowTitle(title + ": Probability Density Function")
        d = common.MATRIX_WINDOW_DEFAULT_SIZE
        w.show_at(100, 100)
        w.resize(d.width() // 3, d.height())
        return w

    @classmethod
    def grids(cls, grids, stats, limits, title, x_axis, y_axis):
        w = cls()
        row = QHBoxLayout()
        row.addWidget(PDFPlot2(grids[0], stats, limits, "Subgroup PDF", x_axis, y_axis))
        row.addWidget(PDFPlot2(grids[1], stats, limits, "Complement PDF", x_axis, y_axis))
        row.addWidget(PDFPlot2(grids[2], stats, limits, "Density Difference", x_axis, y_axis))
        w.content().addLayout(row, 1)
        w.add_button_row([common.build_button("Close", "C", w.close)])
        w.setWindowTitle(title + ": Probability Density Function")
        w.show_at(100, 100, common.MATRIX_WINDOW_DEFAULT_SIZE)
        return w
