"""Pairwise preference matrix of a label ranking.

Values can be fractional, the matrix is full (not a half triangle). All in
float32 like the java float[][].
"""

import math

import numpy as np

from . import log
from .java_compat import to_f32, fdiv, java_round, float_to_string
from .label_ranking import LabelRanking

F = np.float32


class LabelRankingMatrix:
    def __init__(self, size_or_ranking):
        self._mode_matrix = None
        if isinstance(size_or_ranking, LabelRanking):
            r = size_or_ranking
            n = r.get_size()
            self._size = n
            self.matrix = np.zeros((n, n), dtype=np.float32)
            self._mode_matrix = np.zeros((3, n, n), dtype=np.int64)
            for i in range(n):
                for j in range(n):
                    ri, rj = r.get_rank(i), r.get_rank(j)
                    if ri < rj:
                        self.matrix[i, j] = 1
                        self._mode_matrix[2, i, j] = 1
                    elif ri > rj:
                        self.matrix[i, j] = -1
                        self._mode_matrix[0, i, j] = 1
                    else:
                        self._mode_matrix[1, i, j] = 1
        else:
            n = int(size_or_ranking)
            self._size = n
            self.matrix = np.zeros((n, n), dtype=np.float32)

    # java name, other classes use itsMatrix directly
    @property
    def itsMatrix(self):
        return self.matrix

    def clone(self):
        c = LabelRankingMatrix(self._size)
        c.matrix = self.matrix.copy()
        return c

    def get_size(self):
        return self._size

    def add(self, other):
        self.matrix += other.matrix

    def get_mode_matrix(self):
        return self._mode_matrix

    def subtract(self, other):
        self.matrix -= other.matrix

    def divide(self, value):
        self.matrix /= F(value)

    def frobeniuns_norm(self, other):
        f_norm = 0.0
        count = 0
        n = self._size
        for i in range(n):
            for j in range(i + 1, n):
                f_norm += float(other.matrix[i, j]) ** 2
                count += 1
        return fdiv(to_f32(math.sqrt(f_norm)), count)

    def _half_sq_sum(self, other, upper_only=True):
        d = 0.0
        count = 0
        n = self._size
        for i in range(n):
            for j in range(i + 1, n):
                diff = float(F(F(self.matrix[i, j] - other.matrix[i, j]) / F(2)))
                d += diff ** 2
                count += 1
        return d, count

    def norm_distance(self, other):
        d, count = self._half_sq_sum(other)
        return fdiv(to_f32(math.sqrt(d)), count)

    def norm_distance0(self, other):
        """temporary function for tests"""
        d = 1.0
        n = self._size
        for i in range(n):
            for j in range(i + 1, n):
                d *= abs(float(F(F(self.matrix[i, j] - other.matrix[i, j]) / F(2))))
        return to_f32(d)

    def norm_distance_mode(self, other):
        d, count = self._half_sq_sum(other)
        return fdiv(to_f32(math.sqrt(d)), count)

    def wnorm_distance(self, other):
        return to_f32(F(self.norm_distance(other)) * F(self.frobeniuns_norm(other)))

    def _label_distance(self, other, i):
        n = self._size
        dist = F(0)
        for j in range(n):
            if i != j:
                diff = float(F(F(self.matrix[i, j] - other.matrix[i, j]) / F(2)))
                # float += double, java narrows back to float
                dist = F(float(dist) + diff ** 2)
        return fdiv(to_f32(math.sqrt(float(dist))), n - 1)

    def min_distance(self, other):
        # NOTE always >= 0.0 when size > 0
        a_min = float('inf')
        for i in range(self._size):
            a_min = min(a_min, self._label_distance(other, i))
        return a_min

    def labelwise_max(self, other):
        a_max = float('-inf')
        for i in range(self._size):
            a_max = max(a_max, self._label_distance(other, i))
        return a_max

    def labelwise_max0(self, other):
        """temporary function for tests"""
        a_max = float('-inf')
        n = self._size
        for i in range(n):
            dist = F(1)
            for j in range(n):
                if i != j:
                    dist = F(float(dist) * abs(float(F(F(self.matrix[i, j] - other.matrix[i, j]) / F(2)))))
            a_max = max(a_max, float(dist))
        return a_max

    def labelwise_cov(self, other):
        """temporary function for tests"""
        a_max = float('-inf')
        cov = F(0)
        n = self._size
        for i in range(n):
            dist = F(0)
            m1 = F(self.label_mean(other.matrix, i))
            m2 = F(self.label_mean(self.matrix, i))
            for j in range(n):
                if i != j:
                    dist = F(dist + F(F(other.matrix[i, j] - m1) * F(self.matrix[i, j] - m2)))
            cov = F(cov + F(F(-dist) / F(n - 2)))
            a_max = max(a_max, float(cov))
        return a_max

    def pairwise_max(self, other):
        a_max = float('-inf')
        n = self._size
        for i in range(n):
            for j in range(i + 1, n):
                a_max = max(a_max, abs(float(F(F(self.matrix[i, j] - other.matrix[i, j]) / F(2)))))
        return a_max

    def cov_distance(self, other):
        cov = F(0)
        n = self._size
        for i in range(n):
            dist = F(0)
            m1 = F(self.label_mean(other.matrix, i))
            m2 = F(self.label_mean(self.matrix, i))
            for j in range(n):
                if i != j:
                    dist = F(dist + F(F(other.matrix[i, j] - m1) * F(self.matrix[i, j] - m2)))
            with np.errstate(all='ignore'):
                cov = F(cov + F(F(-dist) / F(n - 2)))
        return fdiv(cov, n)

    def label_mean(self, ranking_matrix, label):
        s = F(0)
        for i in range(self._size):
            s = F(s + F(ranking_matrix[label][i]))
        return fdiv(s, self._size - 1)

    def homogeneity(self, other):
        d = F(0)
        n = self._size
        for i in range(n):
            for j in range(i + 1, n):
                d = F(d + abs(F(other.matrix[i, j])))
        # integer division in java
        return fdiv(d, (n * (n + 1)) // 2)

    def get(self, i, j):
        return float(self.matrix[i, j])

    def print(self):
        log.log_command_line("  ================================")
        for i in range(self._size):
            row = LabelRanking.get_letter(i) + " ["
            row += ", ".join(float_to_string(self.matrix[i, j]) for j in range(self._size))
            log.log_command_line(row + "]")
        log.log_command_line("  ================================")

    def find_max(self):
        best = 0.0
        out = None
        n = self._size
        for i in range(n):
            for j in range(i + 1, n):
                v = float(self.matrix[i, j])
                if best < abs(v):
                    rounded = fdiv(java_round(float(F(F(v) * F(1000)))), 1000)
                    out = LabelRanking.get_letter(i) + ">" + LabelRanking.get_letter(j) + " " + float_to_string(rounded)
                    best = abs(v)
        return out

    def find_max_label(self):
        best = float('-inf')
        out = None
        n = self._size
        for i in range(n):
            s = F(0)
            for j in range(n):
                if i != j:
                    s = F(s + abs(F(self.matrix[i, j])))
            s = F(fdiv(s, n - 1))
            if best < float(s):
                # sic: rounds the previous best, not the new one
                rounded = fdiv(java_round(float(F(F(best) * F(1000)))) if math.isfinite(best) else _round_inf(best), 1000)
                out = LabelRanking.get_letter(i) + " (" + float_to_string(rounded) + ")"
                best = float(s)
        return out


def _round_inf(x):
    # Math.round(-Infinity) is Integer.MIN_VALUE for a float argument
    return -2147483648 if x < 0 else 2147483647
