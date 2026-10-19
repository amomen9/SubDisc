"""Convex hulls in 2D, split into an upper (side 0) and lower (side 1) part,
both sorted by x. Only used by the linear BestInterval debug code."""

from . import log
from .java_compat import div, float_to_string, to_f32
from .roc_point import PointFloat


class HullPoint:
    __slots__ = ("x", "y", "label1", "_label2")

    def __init__(self, x, y, label1, label2):
        self.x = int(x)
        self.y = int(y)
        self.label1 = to_f32(label1)
        self._label2 = to_f32(label2)

    @classmethod
    def copy_of(cls, other):
        return cls(other.x, other.y, other.label1, other._label2)

    def get_label2(self):
        return self._label2

    def print(self):
        log.log_command_line("HullPoint (%d,%d) %s, %s" % (self.x, self.y, float_to_string(self.label1), float_to_string(self._label2)))


class ConvexHull:
    def __init__(self, x=None, y=None, label1=None, label2=None):
        """single point hull, or an empty one when called without arguments"""
        self._points = [None, None]
        if x is not None:
            for side in range(2):
                self._points[side] = [HullPoint(x, y, label1, label2)]

    def get_size(self, side):
        return len(self._points[side])

    def get_point(self, side, index):
        return self._points[side][index]

    def _graham_scan_sorted(self):
        """points are sorted by x already, so this is linear"""
        for side in range(2):
            pts = self._points[side]
            n = len(pts)
            if n < 3:
                continue
            sign = 1 if side == 0 else -1
            nxt = [i + 1 for i in range(n)]
            prv = [i - 1 for i in range(n)]
            pruned = 0
            cur = 0
            while nxt[cur] < n and nxt[nxt[cur]] < n:
                p1 = pts[cur]
                p2 = pts[nxt[cur]]
                p3 = pts[nxt[nxt[cur]]]
                # ints are exact in python, no long trick needed
                y21 = p2.y - p1.y
                x32 = p3.x - p2.x
                y32 = p3.y - p2.y
                x21 = p2.x - p1.x
                if sign * y21 * x32 > sign * y32 * x21:   # convex, go on
                    cur = nxt[cur]
                else:   # not convex, drop middle point and step back
                    prv[nxt[nxt[cur]]] = cur
                    nxt[cur] = nxt[nxt[cur]]
                    pruned += 1
                    if cur > 0:
                        cur = prv[cur]
            out = []
            cur = 0
            for _ in range(n - pruned):
                out.append(pts[cur])
                cur = nxt[cur]
            self._points[side] = out

    def concatenate(self, other):
        """assumes this.x < other.x, so the hulls do not overlap"""
        res = ConvexHull()
        for side in range(2):
            res._points[side] = list(self._points[side]) + list(other._points[side])
        res._graham_scan_sorted()
        return res

    def minkowski_difference(self, other, prune_degenerate=True):
        res = ConvexHull()
        neg_inf = float('-inf')
        for side in range(2):
            sign = 1.0 if side == 0 else -1.0
            mine = self._points[side]
            theirs = other._points[1 - side]
            len1 = len(mine)
            len2 = len(theirs)
            hull = []
            i = 0
            j = len2 - 1
            while i < len1 - 1 or j > 0:
                if i == len1 - 1:
                    slope1 = sign * neg_inf   # dummy for last
                else:
                    yd = float(mine[i + 1].y - mine[i].y)
                    xd = float(mine[i + 1].x - mine[i].x)
                    slope1 = div(yd, xd + yd)
                if j == 0:
                    slope2 = sign * neg_inf
                else:
                    yd = float(theirs[j - 1].y - theirs[j].y)
                    xd = float(theirs[j - 1].x - theirs[j].x)
                    slope2 = div(yd, xd + yd)
                if sign * slope1 >= sign * slope2:
                    p = HullPoint.copy_of(mine[i])
                    p._label2 = to_f32(side)
                    hull.append(p)
                    i += 1
                if sign * slope1 <= sign * slope2:
                    p = HullPoint.copy_of(theirs[j])
                    p._label2 = to_f32(1 - side)
                    hull.append(p)
                    j -= 1
            p = HullPoint.copy_of(mine[i])
            p._label2 = to_f32(side)
            hull.append(p)
            p = HullPoint.copy_of(theirs[j])
            p._label2 = to_f32(1 - side)
            hull.append(p)

            size = len(hull)
            new_hull = []
            for k in range(size):
                v = k
                nv = (v + 1) % size
                while hull[v]._label2 == hull[nv]._label2:
                    nv = (nv + 1) % size
                if hull[nv].label1 >= hull[v].label1:
                    v, nv = nv, v
                # own label and the other's label, for intervals the rhs and lhs end points
                new_hull.append(HullPoint(hull[v].x - hull[nv].x, hull[v].y - hull[nv].y, hull[v].label1, hull[nv].label1))
            res._points[side] = new_hull
        if prune_degenerate:
            res._graham_scan_sorted()
        return res


class ConvexHullPoint(PointFloat):
    """like PointFloat, with two labels (old HullPoint)"""

    def __init__(self, x, y, label1, label2):
        PointFloat.__init__(self, x, y)
        self.label1 = to_f32(label1)
        self.label2 = to_f32(label2)

    @classmethod
    def copy_of(cls, other):
        return cls(other.x, other.y, other.label1, other.label2)

    def __str__(self):
        return "ConvexHullPoint[%f, %f] %f, %f" % (self.x, self.y, self.label1, self.label2)

