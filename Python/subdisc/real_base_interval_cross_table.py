"""Michael says: this is basically a copy of NominalCrossTable (for numeric
base intervals, used by the BestInterval algorithm)."""

import numpy as np

from .interval import Interval
from .java_compat import int32

NEG_INF = float('-inf')
POS_INF = float('inf')


class RealBaseIntervalCrossTable:
    def __init__(self, total_count, total_true_positives, domain_copy, value_info):
        self._positive_count = total_true_positives
        self._negative_count = total_count - total_true_positives

        counts = value_info.itsCounts
        tps = value_info.itsTruePositives
        dom = [float(x) for x in domain_copy]
        c_out, tp_out, d_out = [], [], []
        remaining = total_count
        for i in range(len(counts)):
            c = int(counts[i])
            if c == 0:
                continue
            d_out.append(dom[i])
            tp_out.append(int(tps[i]))
            c_out.append(c - int(tps[i]))
            remaining -= c
            if remaining == 0:
                break
        nr_distinct = len(d_out)
        self._use_neg_infty = True
        offset = 1 if self._use_neg_infty else 0
        self._split_point_count = nr_distinct + offset
        self._split_points = [0.0] * self._split_point_count
        nb = self.get_nr_base_intervals()
        self._pos = [0] * nb
        self._neg = [0] * nb
        if self._use_neg_infty:
            self._split_points[0] = NEG_INF
        for k in range(nr_distinct):
            self._split_points[offset + k] = d_out[k]
            self._pos[offset + k] = tp_out[k]
            self._neg[offset + k] = c_out[k]

    def get_split_point(self, index):
        return self._split_points[index]

    def get_base_interval(self, index):
        if self._split_point_count == 0:
            return Interval(NEG_INF, POS_INF)
        if index == 0:
            return Interval(NEG_INF, self._split_points[0])
        if index == self._split_point_count:
            return Interval(self._split_points[index - 1], POS_INF)
        return Interval(self._split_points[index - 1], self._split_points[index])

    def get_positive_count(self, index=None):
        if index is None:
            return self._positive_count
        return self._pos[index]

    def get_negative_count(self, index=None):
        if index is None:
            return self._negative_count
        return self._neg[index]

    def get_nr_split_points(self):
        return self._split_point_count

    def get_nr_base_intervals(self):
        return self._split_point_count + 1

    def get_split_points(self):
        return self._split_points[:self._split_point_count]

    def aggregate_intervals(self):
        """drop split points between base intervals with the same class
        distribution (only for convex quality measures)"""
        prune = 0
        pos, neg, sp = self._pos, self._neg, self._split_points
        start = 1 if self._use_neg_infty else 0
        for i in range(start, self._split_point_count):
            # overflow is no problem, the essential part is ==
            if int32(pos[i] * neg[i + 1]) == int32(pos[i + 1] * neg[i]):
                pos[i - prune] += pos[i + 1]
                neg[i - prune] += neg[i + 1]
                prune += 1
            elif prune > 0:
                pos[i - prune + 1] = pos[i + 1]
                neg[i - prune + 1] = neg[i + 1]
                sp[i - prune] = sp[i]
        self._split_point_count -= prune
