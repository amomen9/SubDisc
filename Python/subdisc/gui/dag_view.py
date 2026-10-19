"""Bayesian network drawing (java DAGView, VisualNode, VisualArc, MShape).

The nodes start in the middle and are spread with a small self organising
map, using a time seeded random, so every window looks a bit different.
Nodes can be dragged with the mouse.
"""

import math
import time

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QWidget

from ..java_compat import JavaRandom

FONT_SIZE = 11
ORANGE = QColor(255, 200, 0)


def _jround(f):
    """Math.round(float)"""
    f = float(np.float32(f))
    if f != f:
        return 0
    if f == float("inf"):
        return 2147483647
    if f == float("-inf"):
        return -2147483648
    return int(math.floor(f + 0.5))


def _fdiv(a, b):
    a = np.float32(a)
    b = np.float32(b)
    with np.errstate(all="ignore"):
        return np.float32(a / b)


class VisualNode:
    def __init__(self, x, y, target):
        self.target = target
        self.x_pos = x
        self.y_pos = y
        self.bounds = QRect()
        self._calc_bounds()

    def _calc_bounds(self):
        w = (len(self.target) + 1) * 7
        h = 2 * FONT_SIZE
        self.bounds = QRect(self.x_pos, self.y_pos, w, h)

    def shift(self, dx, dy):
        self.x_pos += dx
        self.y_pos += dy
        self._calc_bounds()

    def get_x(self):
        return self.bounds.x()

    def get_y(self):
        return self.bounds.y()

    def connect_point(self):
        r = self.bounds
        return QPoint(int(r.x() + 0.5 * r.width()), int(r.y() + 0.5 * r.height()))

    def contains(self, p):
        r = self.bounds
        return r.x() < p.x() < r.x() + r.width() and r.y() < p.y() < r.y() + r.height()

    def paint(self, g):
        r = self.bounds
        # shadow
        g.setPen(Qt.NoPen)
        g.setBrush(QColor(0, 0, 0, 128))
        g.drawRect(r.x() + 3, r.y() + 3, r.width(), r.height())
        # fill and outline
        g.setBrush(ORANGE)
        g.drawRect(r.x(), r.y(), r.width(), r.height())
        g.setPen(QPen(Qt.black, 1.0))
        g.setBrush(Qt.NoBrush)
        g.drawRect(r.x(), r.y(), r.width(), r.height())
        g.drawText(r.x() + 5, r.y() + 4 + FONT_SIZE, self.target)


class VisualArc:
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        self.x1 = self.x2 = self.y1 = self.y2 = 0
        self.calc_bounds()

    def __eq__(self, other):
        return isinstance(other, VisualArc) and self.from_node is other.from_node and self.to_node is other.to_node

    def __hash__(self):
        return hash((id(self.from_node), id(self.to_node)))

    def calc_bounds(self):
        p1a = self.from_node.connect_point()
        p2a = self.to_node.connect_point()
        xmin = min(p1a.x(), p2a.x())
        xmax = max(p1a.x(), p2a.x())
        ymin = min(p1a.y(), p2a.y())
        ymax = max(p1a.y(), p2a.y())
        self._set_bounds(xmin, ymin, xmax - xmin, ymax - ymin)

    def _set_bounds(self, x, y, width, height):
        p1a = self.from_node.connect_point()
        p2a = self.to_node.connect_point()
        if p1a.x() <= p2a.x():
            r1, r2 = self.from_node.bounds, self.to_node.bounds
        else:
            r1, r2 = self.to_node.bounds, self.from_node.bounds
        slope = _fdiv(height, width)   # could be /0
        F = np.float32
        two = F(2)

        def gt(r):
            return _fdiv(r.height(), r.width()) > slope

        if (p1a.x() > p2a.x() and p1a.y() > p2a.y()) or (p1a.x() < p2a.x() and p1a.y() < p2a.y()):
            # topleft to bottomright
            if gt(r1):
                self.x1 = x + _jround(_fdiv(r1.width(), two))
                self.y1 = y + _jround(_fdiv(F(slope * F(r1.width())), two))
            else:
                self.x1 = x + _jround(_fdiv(_fdiv(r1.height(), slope), two))
                self.y1 = y + _jround(_fdiv(r1.height(), two))
            if gt(r2):
                self.x2 = x + width - _jround(_fdiv(r2.width(), two))
                self.y2 = y + height - _jround(_fdiv(F(slope * F(r2.width())), two))
            else:
                self.x2 = x + width - _jround(_fdiv(_fdiv(r2.height(), slope), two))
                self.y2 = y + height - _jround(_fdiv(r2.height(), two))
        else:
            # topright to bottomleft
            if gt(r1):
                self.x2 = x + _jround(_fdiv(r1.width(), two))
                self.y1 = y + _jround(_fdiv(F(slope * F(r1.width())), two))
            else:
                self.x2 = x + _jround(_fdiv(_fdiv(r1.height(), slope), two))
                self.y1 = y + _jround(_fdiv(r1.height(), two))
            if gt(r2):
                self.x1 = x + width - _jround(_fdiv(r2.width(), two))
                self.y2 = y + height - _jround(_fdiv(F(slope * F(r2.width())), two))
            else:
                self.x1 = x + width - _jround(_fdiv(_fdiv(r2.height(), slope), two))
                self.y2 = y + height - _jround(_fdiv(r2.height(), two))

    def paint(self, g):
        g.setPen(QPen(Qt.black, 1.5))
        g.drawLine(self.x1, self.y1, self.x2, self.y2)
        # arrow head
        size = 15
        phi = math.radians(20)
        p1a = self.from_node.connect_point()
        p2a = self.to_node.connect_point()
        theta = math.atan2(self.y1 - self.y2, self.x1 - self.x2)
        if p1a.y() <= p2a.y():
            tx, ty = self.x2, self.y2
        else:
            theta += math.pi
            tx, ty = self.x1, self.y1
        poly = QPolygon([QPoint(tx, ty),
                         QPoint(tx + int(size * math.cos(theta - phi)), ty + int(size * math.sin(theta - phi))),
                         QPoint(tx + int(size * math.cos(theta + phi)), ty + int(size * math.sin(theta + phi))),
                         QPoint(tx, ty)])
        g.setPen(Qt.NoPen)
        g.setBrush(QBrush(Qt.black))
        g.drawPolygon(poly)


class DAGView(QWidget):
    def __init__(self, dag):
        super().__init__()
        self._dag = dag
        self._size = dag.get_size()
        self._random = JavaRandom(int(time.time() * 1000))   # truly random
        self._nodes = []
        self._arcs = []
        self._width = 0
        self._height = 0
        self._drag = None
        self._drag_point = QPoint()
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(p)
        self.setMouseTracking(False)

    def set_dag_area(self, width, height):
        self._width = width
        self._height = height

    def draw_dag(self):
        for i in range(self._size):
            x = self._width // 2 + self._random.next_int(20)
            y = self._height // 2 + self._random.next_int(20)
            self._nodes.append(VisualNode(x, y, self._dag.get_name(i)))
        self._connect_all()
        self.apply_som()
        self.update_connectors()

    def draw_dag_like(self, other):
        """use the layout of another DAGView"""
        self._nodes = other._nodes   # no deep copy
        self._connect_all()
        self.update_connectors()

    def _connect_all(self):
        for i in range(self._size):
            for j in range(i):
                c = self._dag.get_node(j).is_connected(i)
                if c == 1:      # i-->j
                    self.connect(i, j)
                elif c == 2:    # i<--j
                    self.connect(j, i)

    def apply_som(self):
        if self._size == 0:
            return
        F = np.float32
        nb = F(1.0)
        while nb > F(0.0):
            for _ in range(1000):
                x = self._random.next_int(self._width - 30) - 50
                y = self._random.next_int(self._height - 20) - 50
                winner = 0
                v = self._nodes[0]
                best = float((v.get_x() - x) * (v.get_x() - x) + (v.get_y() - y) * (v.get_y() - y))
                for j in range(1, self._size):
                    v = self._nodes[j]
                    d = float((v.get_x() - x) * (v.get_x() - x) + (v.get_y() - y) * (v.get_y() - y))
                    if d < best:
                        best = d
                        winner = j
                v = self._nodes[winner]
                node = self._dag.get_node(winner)
                v.shift(int(F(0.1) * F(x - v.get_x())), int(F(0.1) * F(y - v.get_y())))
                f = F(nb * F(0.1))
                for j in range(self._size):
                    if node.is_connected(j) > 0:
                        v = self._nodes[j]
                        v.shift(int(F(f * F(x - v.get_x()))), int(F(f * F(y - v.get_y()))))
            nb = F(nb - F(0.05))

    def update_connectors(self):
        for a in self._arcs:
            a.calc_bounds()

    def connect(self, id1, id2):
        a = VisualArc(self._nodes[id1], self._nodes[id2])
        if a not in self._arcs:
            self._arcs.append(a)

    def paintEvent(self, _e):
        g = QPainter(self)
        g.setRenderHint(QPainter.Antialiasing, True)
        f = QFont(self.font())
        f.setPixelSize(FONT_SIZE)
        g.setFont(f)
        for a in self._arcs:
            a.paint(g)
        for n in self._nodes:
            n.paint(g)
        g.end()

    def mousePressEvent(self, e):
        p = e.position().toPoint()
        self._drag = None
        for n in reversed(self._nodes):
            if n.contains(p):
                self._drag = n
                self._drag_point = QPoint(p.x() - n.bounds.x(), p.y() - n.bounds.y())
                break

    def mouseMoveEvent(self, e):
        if self._drag is None:
            return
        p = e.position().toPoint()
        r = self._drag.bounds
        self._drag.bounds = QRect(p.x() - self._drag_point.x(), p.y() - self._drag_point.y(), r.width(), r.height())
        self.update_connectors()
        self.update()

    def mouseReleaseEvent(self, _e):
        self._drag = None
        self.update_connectors()
        self.update()
