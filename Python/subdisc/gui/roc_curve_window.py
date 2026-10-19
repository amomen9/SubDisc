"""ROC curve of a nominal result: the convex hull, a point per subgroup,
the minimum / maximum coverage lines and the quality measure isometrics in
the background (green positive, red negative)."""

import math
from decimal import Decimal

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QAbstractItemView, QTableWidget, QTableWidgetItem, QWidget

from .. import log
from .. import ui_hooks
from ..file_type import FileType
from ..java_compat import double_to_string
from ..qm import QM
from ..roc_point import SubgroupROCPoint
from ..subgroup_set import SubgroupSet
from ..target_type import TargetType
from . import common

F = np.float32
RESOLUTION = 400


class ROCCurve(QWidget):
    def __init__(self, subgroup_set, search_parameters, quality_measure):
        super().__init__()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)
        self.setMinimumSize(300, 300)
        self._ok = False
        self._auc = ""
        roc = subgroup_set.get_roc_list()
        if roc is None:
            return
        self._ok = True
        self._auc = common.number(roc.get_area_under_curve(), 3)
        self._curve = [(0.0, 0.0)] + [(float(p.get_fpr()), -float(p.get_tpr())) for p in roc] + [(1.0, -1.0)]
        self._points = [(float(pt.get_fpr()), -float(pt.get_tpr())) for pt in (SubgroupROCPoint(s) for s in subgroup_set)]

        total = subgroup_set.get_total_coverage()
        self._total_target = F(subgroup_set.get_total_target_coverage())
        min_cov = search_parameters.get_minimum_coverage()
        max_cov = int(F(F(total) * F(search_parameters.get_maximum_coverage_fraction())))
        false_cov = F(F(total) - self._total_target)
        self._min, self._max = min_cov, max_cov
        self._qm = quality_measure
        x_min = F(F(min_cov) / false_cov)
        y_min = F(F(min_cov) / self._total_target)
        x_max = F(F(max_cov) / false_cov)
        y_max = F(F(max_cov) / self._total_target)
        x_start = F(F(y_max - F(1)) * F(x_max / y_max))
        y_start = F(1)
        if y_max < 1:   # crosses the left boundary, not the top
            x_start = F(0)
            y_start = y_max
        x_end = F(1)
        y_end = F(y_max - F(y_max / x_max))
        if x_max < 1:   # crosses the bottom boundary, not the right one
            x_end = x_max
            y_end = F(0)
        self._x_min, self._x_end, self._y_end = float(x_min), float(x_end), float(y_end)
        L = []
        L.append(((float(x_min), 0.0), (0.0, -float(y_min))))                       # min cov
        L.append(((float(x_start), -float(y_start)), (float(x_end), -float(y_end))))  # max cov
        self._min_support = search_parameters.get_minimum_support()
        if self._min_support > 0:
            y = -float(F(F(self._min_support) / self._total_target))
            L.append(((0.0, y), (1.0, y)))
        L += [((0, 0), (0, -1)), ((0, -1), (1, -1)), ((1, -1), (1, 0)), ((1, 0), (0, 0))]
        for i in range(11):
            t = float(F(i) * F(0.1))
            L.append(((t, 0.0), (t, 0.01)))
            L.append(((0.0, -t), (-0.01, -t)))
        self._lines = L
        self._isometrics = None

    def get_area_under_curve(self):
        return self._auc

    def _make_isometrics(self):
        """the colours do not depend on the window size, compute them once"""
        qm = self._qm
        N = qm.get_nr_records()
        p = qm.get_nr_positives()
        a_max = max(qm.calculate(p, p), qm.calculate(p, N))
        a_max = float(F(max(a_max, qm.calculate(0, N - p))))
        img = QImage(RESOLUTION, RESOLUTION, QImage.Format_RGB32)
        cache = {}
        for i in range(RESOLUTION):
            x = F(F(i) / F(RESOLUTION))
            neg = F(x * F(N - p))   # can be fractional, the picture looks nicer
            for j in range(RESOLUTION):
                y = F(-F(j + 1) / F(RESOLUTION))
                pos = F(-y * F(p))
                key = (int(pos), max(1, int(F(neg + pos))))
                q = cache.get(key)
                if q is None:
                    try:
                        q = qm.calculate(key[0], key[1])
                    except Exception:
                        q = 0.0
                    cache[key] = q
                with np.errstate(all="ignore"):
                    v = 255.0 * q / a_max
                v = int(v) if math.isfinite(v) else (0 if v != v else (2147483647 if v > 0 else -2147483648))
                negative = v < 0
                if negative:
                    v = -v
                v = max(min(v, 255), 0)
                v = 15 * (v // 15)
                c = QColor(255, 255 - v, 255 - v) if negative else QColor(255 - v, 255, 255 - v)
                img.setPixelColor(i, RESOLUTION - 1 - j, c)
        self._isometrics = img

    def paintEvent(self, _e):
        if not self._ok:
            return
        if self._isometrics is None:
            common.set_busy(True)
            try:
                self._make_isometrics()
            finally:
                common.set_busy(False)
        s = max(260.0, min(self.width(), self.height()) * 0.85)

        def P(u, v):
            return QPointF((u + 0.15) * s, (v + 1.05) * s)

        g = QPainter(self)
        g.drawImage(QRectF(P(0, -1), P(1, 0)), self._isometrics)
        # subgroups
        g.setPen(QPen(Qt.black, 3.0))
        for u, v in self._points:
            g.drawPoint(P(u, v))
        g.setRenderHint(QPainter.Antialiasing, True)
        path = QPainterPath(P(*self._curve[0]))
        for u, v in self._curve[1:]:
            path.lineTo(P(u, v))
        g.setPen(QPen(Qt.black, 2.0))
        g.drawPath(path)
        g.setPen(QPen(Qt.black, 1.0))
        for (u1, v1), (u2, v2) in self._lines:
            g.drawLine(P(u1, v1), P(u2, v2))

        f = QFont(self.font())
        f.setPixelSize(11)
        g.setFont(f)
        g.drawText(P(-0.05, 0.04), "(0,0)")
        g.drawText(P(0.44, 0.08), "FPR")
        g.drawText(P(-0.1, -0.44), "TPR")
        for i in range(1, 11):
            t = float(F(i) * F(0.1))
            txt = common.number(t, 1)
            g.drawText(P(t, 0.04), txt)
            g.drawText(P(-0.07, -t), txt)
        f.setPixelSize(10)
        g.setFont(f)
        g.drawText(P(0.02, -0.96), common.number(self._qm.get_roc_heaven(), 4))
        g.drawText(P(0.9, -0.02), common.number(self._qm.get_roc_hell(), 4))
        g.drawText(P(self._x_min, -0.03), str(self._min))
        g.drawText(P(self._x_end + 0.01, -max(self._y_end, 0.03)), str(self._max))
        if self._min_support > 0:
            g.drawText(P(1 + 0.01, -float(F(self._min_support) / self._total_target)), str(self._min_support))
        g.end()


def _plain(d):
    """DecimalFormat("0") with all fraction digits, always a '.'"""
    d = float(d)
    if math.isnan(d):
        return "NaN"
    if math.isinf(d):
        return "∞" if d > 0 else "-∞"
    s = format(Decimal(repr(d)), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("-0", ""):
        s = "-0" if s == "-0" else "0"
    return s


class ROCCurveWindow(common.Window):
    def __init__(self, subgroup_set, search_parameters, quality_measure):
        super().__init__()
        self._set = subgroup_set
        self._sp = search_parameters
        curve = ROCCurve(subgroup_set, search_parameters, quality_measure)
        rows = subgroup_set.get_roc_list_subgroups() if curve._ok else []
        t = QTableWidget(len(rows), len(SubgroupSet.ROC_HEADER))
        t.setHorizontalHeaderLabels(list(SubgroupSet.ROC_HEADER))
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.verticalHeader().setVisible(False)
        for r, row in enumerate(rows):
            for c, v in enumerate(row):
                t.setItem(r, c, QTableWidgetItem(double_to_string(v) if isinstance(v, float) else str(v)))
        t.horizontalHeader().setStretchLastSection(True)
        t.setMaximumHeight(110)
        self.content().addWidget(t)
        self.content().addWidget(curve, 1)
        self.add_button_row([common.build_button("GnuPlot", "G", self._gnuplot),
                             common.build_button("Close", "C", self.close)])
        self.setWindowTitle("ROC Curve (area under curve: " + curve.get_area_under_curve() + ")")
        self.show_at(100, 100, common.ROC_WINDOW_DEFAULT_SIZE)

    def _gnuplot(self):
        path = ui_hooks.choose_save_file(FileType.PLT)
        if path is None:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as w:
                w.write(self._gnuplot_string())
        except OSError:
            log.log_command_line("Error while writing: " + str(path))

    def _gnuplot_string(self):
        sz = _plain(0.001)
        fmt_sg = "set object circle at %s, %s, %d size " + sz + " fc rgb \"black\"\n"
        fmt_hull = "set arrow from %s, %s, %d to %s, %s, %d nohead lt 1 lw 2 lc rgb \"black\"\n"
        roc = self._set.get_roc_list()
        auc = _plain(roc.get_area_under_curve()).replace(".", "p")
        fmt_file = "roc-%s-auc" + auc + ".eps"
        sb = []
        a = sb.append
        a("# general plot settings\n")
        a("set xlabel \"fpr\"\n")
        a("set ylabel \"tpr\"\n")
        a("set xrange [0:1]\n")
        a("set yrange [0:1]\n")
        a("set xtics 0.1\n")
        a("set ytics 0.1\n")
        a("\n")
        a("# basic functions for Quality Measures\n")
        a("N = " + str(self._set.get_total_coverage()) + ".0\n")
        a("p = " + str(int(self._set.get_total_target_coverage())) + ".0\n")
        a("n = N-p\n")
        a("pos(y) = y*p\n")
        a("neg(x) = x*(N-p)\n")
        a("aTotalTargetCoverageNotBody(y) = p-pos(y)\n")
        a("aCountNotHeadNotBody(x) = N-(p+neg(x))\n")
        a("aCountBody(x,y) = neg(x)+pos(y)\n")
        a("aCountNotHeadBody(x,y) = aCountBody(x,y) - pos(y)\n")
        a("sqr(x) = x*x\n")
        a("max(x,y) = x>y?x:y\n")
        a("expect(x,y,z) = x*(y/x)*(z/x)\n")
        a("e11(x,y,z) = expect(x,y,z)\n")
        a("e01(x,y,z) = expect(x,x-y,z)\n")
        a("e10(x,y,z) = expect(x,y,x-z)\n")
        a("e00(x,y,z) = expect(x,x-y,x-z)\n")
        a("lg(x) = log(x)/log(2)\n")
        a("H(x) = -x*lg(x)-(1-x)*lg(1-x)\n")
        a("mi(x,y,z) = x*log(x/((x+y)*(x+z)))\n")
        a("\n")
        a("# definitions of implmented Quality Measures\n")
        for q in QM.get_quality_measures(TargetType.SINGLE_NOMINAL):
            a(QM.get_definition(q) + "\n")
        a("\n")
        a("# a point for every subgroup\n")
        for s in self._set:
            a(fmt_sg % (_plain(s.get_false_positive_rate()), _plain(s.get_true_positive_rate()), s.get_id()))
        a("\n")
        a("# points/lines for subgroups on the convex hull\n")
        ox, oy, oz = 0.0, 0.0, 0
        for pt in roc:
            nx, ny, nz = pt.get_fpr(), pt.get_tpr(), pt.ID
            a(fmt_hull % (_plain(ox), _plain(oy), oz, _plain(nx), _plain(ny), nz))
            ox, oy, oz = nx, ny, nz
        a(fmt_hull % (_plain(ox), _plain(oy), oz, _plain(1.0), _plain(1.0), 0))
        a("\n")
        a("# Quality Measure isometrics\n")
        a("# if the 'splot' command below does not generate any output\n")
        a("# check that the function it calls is included in the\n")
        a("# list of Quality Measure function definitions above\n")
        name = self._sp.get_quality_measure().GUI_TEXT.replace(" ", "_").replace("-", "_")
        a("splot %s(x,y) lt 3 lw 0.5\n" % name)
        a("\n")
        a("# settings for Quality Measure isometrics\n")
        a("set isosamples 100\n")
        a("set contour\n")
        a("set cntrparam cubicspline\n")
        a("set cntrparam order 5\n")
        a("set cntrparam points 20\n")
        a("set cntrparam levels 20\n")
        a("unset clabel\n")
        a("set view map\n")
        a("unset surface\n")
        a("\n")
        a("set terminal postscript eps size 5,5\n")
        a("set output \"" + (fmt_file % name) + "\"\n")
        a("replot\n")
        a("set output\n")
        a("set terminal pop\n")
        a("set size 1,1\n")
        a("\n")
        return "".join(sb)
