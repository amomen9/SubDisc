"""Naive ROC convex hull, used by the ROC_BEAM search strategy."""

import threading

from .candidate import Candidate
from .java_compat import div, float_compare
from .roc_point import PointDouble
from .tree_set import TreeSet

ZERO_ZERO = PointDouble(0.0, 0.0)
ONE_ONE = PointDouble(1.0, 1.0)

_debug_max_size = 0


def _slope(a, b):
    return div(b.y - a.y, b.x - a.x)


class ConvexHullROCNaive:
    def __init__(self, subgroup_set=None):
        self.hull = []
        self._lock = threading.RLock()
        if subgroup_set is not None:
            from .roc_point import SubgroupROCPoint
            for s in subgroup_set:
                self.add(SubgroupROCPoint(s))

    def add(self, p):
        """True when the hull changed."""
        global _debug_max_size
        with self._lock:
            hull = self.hull
            if len(hull) > _debug_max_size:
                _debug_max_size = len(hull)
            # always under the hull
            if p.y < p.x:
                return False
            i = len(hull)
            if i == 0:
                hull.append(p)
                return True
            # find insertion index, looping back simplifies == handling
            i -= 1
            while i >= 0:
                q = hull[i]
                if p.x < q.x:
                    i -= 1
                    continue
                elif p.x == q.x:
                    # special, (multiple) points on the TPR-axis
                    if p.x == 0.0:
                        if p.y < q.y:
                            i -= 1
                            continue
                        elif p.y == q.y:
                            hull.insert(i + 1, p)
                            return True
                        else:
                            i += 1
                            hull.insert(i, p)
                            break
                    if p.y < q.y:
                        return False
                    elif p.y == q.y:
                        hull.insert(i + 1, p)
                        return True
                    else:
                        hull[i] = p
                        break
                else:
                    if self._below(q, p, i):
                        return False
                    i += 1
                    hull.insert(i, p)
                    break
            if i == -1:
                # smallest x, put it in front
                if p.x == 0.0:
                    i += 1
                    hull.insert(i, p)
                elif self._below(ZERO_ZERO, p, 0):
                    return False
                else:
                    i += 1
                    hull.insert(i, p)

            # update the whole hull (simple, clean)
            left = ZERO_ZERO
            im, ir = 0, 1
            while im < len(hull):
                m = hull[im]
                r = ONE_ONE if ir == len(hull) else hull[ir]
                if left.x == m.x:
                    left = m
                    im += 1
                    ir += 1
                    continue
                # do not remove any points on the top line
                if m.y == 1.0:
                    break
                if m.x == r.x and m.y == r.y:
                    ir += 1
                    continue
                lm = _slope(left, m)
                lr = _slope(left, r)
                if lm < lr:
                    del hull[im:ir]
                    left = ZERO_ZERO
                    im, ir = 0, 1
                else:
                    if r is ONE_ONE:
                        break
                    left = m
                    im += 1
                    ir = im + 1
            return True

    def _below(self, left, p, index):
        right = ONE_ONE if index == len(self.hull) - 1 else self.hull[index + 1]
        if p.y >= right.y:
            return False
        return _slope(left, p) < _slope(left, right)

    def size(self):
        with self._lock:
            return len(self.hull)

    def to_tree_set(self):
        candidates = TreeSet()
        with self._lock:
            for p in self.hull:
                candidates.add(Candidate(p.get_subgroup()))
        return candidates

    def sort(self):
        """FPR, then TPR, then subgroup ID (call after SubgroupSet.set_ids())"""
        from functools import cmp_to_key

        def cmp(a, b):
            if a is b:
                return 0
            c = float_compare(a.x, b.x)
            if c != 0:
                return c
            c = float_compare(a.y, b.y)
            return c if c != 0 else a.ID - b.ID

        with self._lock:
            self.hull.sort(key=cmp_to_key(cmp))
            return self

    def debug(self):
        with self._lock:
            size = len(self.hull)
            print("size = %d max_size = %d" % (size, _debug_max_size))
            if size == 0:
                _debug_pair(ZERO_ZERO, ONE_ONE)
                return
            _debug_pair(ZERO_ZERO, self.hull[0])
            for i in range(1, size):
                _debug_pair(self.hull[i - 1], self.hull[i])
            _debug_pair(self.hull[-1], ONE_ONE)
        print()

    def remove_pure_subgroups(self):
        """XXX MM (java) last minute hack: drop all points on the TPR-axis."""
        if not self.hull or self.hull[0].x != 0.0:
            return
        for i, p in enumerate(self.hull):
            if p.x != 0.0:
                del self.hull[:i]
                return


def _debug_pair(p, q):
    from .java_compat import double_to_string
    print(q)
    print("\t" + double_to_string(_slope(p, q)))
