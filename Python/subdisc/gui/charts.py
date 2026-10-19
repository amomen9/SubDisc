"""matplotlib in Qt, stands in for the JFreeChart ChartPanel (zoom, pan and
save as image come with the toolbar)."""

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT   # noqa: E402
from matplotlib.figure import Figure   # noqa: E402
from PySide6.QtWidgets import QVBoxLayout, QWidget   # noqa: E402


class ChartPanel(QWidget):
    def __init__(self, parent=None, toolbar=True, width=8, height=5):
        super().__init__(parent)
        self.figure = Figure(figsize=(width, height), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        if toolbar:
            self.toolbar = NavigationToolbar2QT(self.canvas, self)
            lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        self.ax = self.figure.add_subplot(111)
        style_axes(self.ax)

    def redraw(self):
        self.canvas.draw_idle()


def style_axes(ax):
    """white plot with gray grid lines, like the java charts"""
    ax.set_facecolor("white")
    ax.grid(True, color="#a0a0a0", linewidth=0.5, linestyle="-")
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color("#808080")
