"""CAUCWindow (ROC curves for a range of thresholds on a numeric target) and
the Gnuplot writer it uses. Only reachable through Process.CAUC_HEAVY,
which is off by default."""

import time

from .. import log
from ..java_compat import format_fixed
from . import common
from .charts import ChartPanel

DELIMITER = "\t"
COLUMN_HEADER = "threshold" + DELIMITER + "FPR" + DELIMITER + "TPR"
DATA_EXT = ".dat"
SCRIPT_EXT = ".gp"


class CAUCWindow(common.Window):
    def __init__(self, target, statistics):
        super().__init__()
        self._column = target
        self._statistics = statistics
        if target is None or statistics is None:
            return
        chart = ChartPanel()
        ax = chart.ax
        for lst in statistics:
            pts = {}
            # x -> y, an existing x is updated (addOrUpdate), order of insertion
            i = 3
            while i + 1 < len(lst):
                pts[lst[i]] = lst[i + 1]
                i += 2
            ax.plot(list(pts.keys()), list(pts.values()), linewidth=1.0, label=_make_title(lst))
        self.content().addWidget(chart, 1)
        self.add_button_row([common.build_button("GnuSplot 3D", "S", lambda: write_splot_script(self._column, self._statistics)),
                             common.build_button("Close", "C", self.close)])
        self.setWindowTitle("CAUC Window " + target.get_name())
        self.show_at(50, 50, common.WINDOW_DEFAULT_SIZE)


def _make_title(stats):
    return "t=%s n=%s auc=%s" % (common.number(stats[0], 2), common.number(stats[1], 2), common.number(stats[2], 2))


# ----------------------------------------------------------------------
# Gnuplot
# ----------------------------------------------------------------------

def write_splot_script(column, statistics):
    base = "%s_%d" % (column.get_name(), int(time.time() * 1000))
    _write_data(base, statistics)
    _write_splot_script(column.get_name(), base, statistics)


def _f(x):
    return format_fixed(float(x), 6)


def _write_data(base, statistics):
    path = base + DATA_EXT
    try:
        with open(path, "w", encoding="utf-8", newline="") as w:
            n = len(statistics)
            w.write("#" + "1" + "".join(DELIMITER + str(i) for i in range(2, n * 3 + 1)) + "\n")
            w.write("#" + (DELIMITER * 3).join(_stats_title(s) for s in statistics) + "\n")
            w.write("#" + DELIMITER.join([COLUMN_HEADER] * n) + "\n")
            mx = max(len(s) for s in statistics)
            for i in range(3, mx, 2):
                w.write(DELIMITER.join(_datum(i, s) for s in statistics) + "\n")
        _log("data", path)
    except OSError:
        pass


def _stats_title(s):
    return "t=%s_n=%s_auc=%s" % (_f(s[0]), _f(s[1]), _f(s[2]))


def _datum(i, s):
    # NOTE (java) the script could use the less portable 'set datafile missing'
    if len(s) <= i:
        return DELIMITER.join((_f(s[0]), _f(1.0), _f(1.0)))
    return DELIMITER.join((_f(s[0]), _f(s[i]), _f(s[i + 1])))


def _write_splot_script(title, base, statistics):
    path = base + SCRIPT_EXT
    try:
        with open(path, "w", encoding="utf-8", newline="") as w:
            n = len(statistics)
            w.write(PLOT_CODE % (base + DATA_EXT, 1, n * 3, _f(statistics[0][0]), _f(statistics[n - 1][0]), title))
        _log("script", path)
    except OSError:
        pass


def _log(kind, path):
    import os
    log.log_command_line("Gnuplot %s written: '%s'" % (kind, os.path.abspath(path)))


PLOT_CODE = (
    "##### DECLARATION OF DEFAULTS #####\n"
    "\n"
    "INPUT_FILE = '%s'\n"
    "SPLOT = %d\n"
    "OUTPUT_FILE = SPLOT ? INPUT_FILE.'.3d.' : INPUT_FILE.'.2d.'\n"
    "\n"
    "NR_COLUMNS = %d\n"
    "THRESHOLD_MIN = %s\n"
    "THRESHOLD_MAX = %s\n"
    "\n"
    "TITLE = '%s'\n"
    "\n"
    "FONT = 'Helvetica'\n"
    "FONT_SIZE = 14\n"
    "\n"
    "XLABEL = SPLOT ? 'Threshold' : 'FPR'\n"
    "YLABEL = SPLOT ? 'FPR' : 'TPR'\n"
    "ZLABEL = 'TPR'\n"
    "\n"
    "TIC_SIZE = 0.2\n"
    "USE_MINOR_TICS = 0\n"
    "\n"
    "LINE_STYLE = 1\n"
    "LINE_WIDTH = 1\n"
    "\n"
    "\n"
    "\n"
    "##### SETUP #####\n"
    "\n"
    "set terminal postscript eps enhanced FONT FONT_SIZE\n"
    "set output OUTPUT_FILE.'eps'\n"
    "\n"
    "set title TITLE\n"
    "\n"
    "set xlabel XLABEL\n"
    "set ylabel YLABEL\n"
    "set zlabel ZLABEL\n"
    "\n"
    "if (!SPLOT) set xtics 0 TIC_SIZE\n"
    "set ytics 0 TIC_SIZE\n"
    "set ztics 0 TIC_SIZE\n"
    "\n"
    "# NOTE will also set mxtics\n"
    "if (USE_MINOR_TICS) set mxtics; set mytics; set mztics;\n"
    "\n"
    "if (!SPLOT) set xrange [0 : 1]; else set xrange [THRESHOLD_MIN : THRESHOLD_MAX]\n"
    "set yrange [0 : 1]\n"
    "set zrange [0 : 1]\n"
    "\n"
    "# forces ground plain to height 0.0\n"
    "set ticslevel 0\n"
    "\n"
    "\n"
    "\n"
    "##### PLOTTING #####\n"
    "\n"
    "if (SPLOT) \\\n"
    "\tsplot for [i=1 : NR_COLUMNS-2 : 3] \\\n"
    "\t\tINPUT_FILE u i : i+1 : i+2 w l ls LINE_STYLE lw LINE_WIDTH notitle; \\\n"
    "else \\\n"
    "\tplot for [i=1 : NR_COLUMNS-2 : 3] \\\n"
    "\t\tINPUT_FILE u i+1 : i+2 w l ls LINE_STYLE lw LINE_WIDTH notitle\n"
    "\n"
    "#set terminal pslatex\n"
    "#set output OUTPUT_FILE.'tex'\n"
    "#replot\n"
    "\n"
    "#set terminal postscript landscape enhanced FONT 8\n"
    "#set output OUTPUT_FILE.'ps'\n"
    "#replot\n"
    "\n"
    "#set terminal postscript eps FONT FONT_SZIE\n"
    "#set output OUTPUT_FILE.'eps'\n"
    "#replot\n"
    "\n"
    "#set terminal svg\n"
    "#set output OUTPUT_FILE.'svg'\n"
    "\n"
    "#replot\n"
    "set output\n"
    "#set terminal windows\n"
    "#platform-independent way of restoring terminal by push/pop\n"
    "set terminal pop\n"
    "#set size 1,1\n\n")

# takes 3 String parameters in order: data_file_name, x_label, y_label
PLOT_CODE_PDF_2D = (
    "##### PUT THE NAME OF THE VARIABLES HERE #####\n"
    "DATA_FILE='%s'\n"
    "X_LABEL='%s'\n"
    "Y_LABEL='%s'\n"
    "\n"
    "##### DO NOT CHANGE ANYTING BELOW #####\n"
    "PWD=\"`pwd`/\"\n"
    "INPUT_FILE=PWD.DATA_FILE\n"
    "TITLE=sprintf(\"PDF for: %%s versus %%s.\", X_LABEL, Y_LABEL)\n"
    "\n"
    "set terminal postscript eps enhanced 'Helvetica' 14\n"
    "set output INPUT_FILE.'.eps'\n"
    "\n"
    "#set encoding iso_8859_1\n"
    "#set title TITLE\n"
    "\n"
    "# set key spacing 1 when outputting to .(e)ps\n"
    "set key spacing 1\n"
    "set xlabel X_LABEL\n"
    "set ylabel Y_LABEL\n"
    "set zlabel 'KDE'\n"
    "\n"
    "set xtics\n"
    "set ytics\n"
    "set ztics\n"
    "\n"
    "set zrange [-1:1]\n"
    "\n"
    "set view 60,30,.8\n"
    "unset colorbox\n"
    "set hidden3d\n"
    "\n"
    "set pm3d\n"
    "unset surface\n"
    "set contour\n"
    "\n"
    "unset key\n"
    "\n"
    "splot INPUT_FILE using 1:2:3 with line\n"
    "\n"
    "set output\n"
    "# platform-independent way of restoring terminal by push/pop\n"
    "set terminal pop\n\n")

PLOT_CODE_PDF_2D_HEATMAP = (
    "##### PUT THE NAME OF THE VARIABLES HERE #####\n"
    "DATA_FILE='%s'\n"
    "X_LABEL='%s'\n"
    "Y_LABEL='%s'\n"
    "\n"
    "##### DO NOT CHANGE ANYTING BELOW #####\n"
    "PWD=\"`pwd`/\"\n"
    "INPUT_FILE=PWD.DATA_FILE\n"
    "TITLE=sprintf(\"Heatmap for: %%s versus %%s.\", X_LABEL, Y_LABEL)\n"
    "\n"
    "set terminal postscript eps enhanced 'Helvetica' 14\n"
    "set output INPUT_FILE.'.heatmap.eps'\n"
    "\n"
    "#set encoding iso_8859_1\n"
    "#set title TITLE\n"
    "\n"
    "# set key spacing 1 when outputting to .(e)ps\n"
    "set key spacing 1\n"
    "set xlabel X_LABEL\n"
    "set ylabel Y_LABEL\n"
    "set cblabel 'KDE'\n"
    "\n"
    "set xtics scale 0,0\n"
    "set ytics scale 0,0\n"
    "#set nocbtics\n"
    "\n"
    "#set style data lines\n"
    "# range to large for most settings - picture will be single color only\n"
    "#set cbrange [ -1.0 : 1.0 ] noreverse nowriteback\n"
    "\n"
    "#set view map scale 1\n"
    "set view map\n"
    "set colorbox\n"
    "unset contour\n"
    "\n"
    "# use splot and pm3d to allow for smoothing in eps\n"
    "set pm3d map\n"
    "# set the level of smoothing (0,0 lets gnuplot decide)\n"
    "set pm3d interpolate 0,0\n"
    "\n"
    "# palette useful only for color terminals (svg) - red/green instead of default\n"
    "#set palette rgbformulae -7, 2, -7\n"
    "\n"
    "unset key\n"
    "\n"
    "# smoothed image - for pixelated image use 'splot INPUT_FILE with image'\n"
    "splot INPUT_FILE\n"
    "\n"
    "set output\n"
    "# platform-independent way of restoring terminal by push/pop\n"
    "set terminal pop\n\n")
