"""n-dimensional cube of counts (java CrossCube extends DataCube).
TODO (java) overflow when dimensions > 31."""

import math

import numpy as np

from . import log
from . import function


class DataCube:
    def __init__(self):
        self._size = 0

    def get_size(self):
        return self._size

    def set_size(self, size):
        self._size = size


class CrossCube(DataCube):
    def __init__(self, dimensions):
        DataCube.__init__(self)
        power = int(math.pow(2, dimensions))
        self.set_size(power)
        self._dimensions = dimensions
        self._counts = np.zeros(power, dtype=np.int64)
        self._total_count = 0

    def increment_count(self, bitset):
        """bitset: an ItemSet or a sequence of booleans"""
        self._increment(self._get_index(bitset))

    def _increment(self, index):
        if index >= self._size:
            log.error("out of bounds")
        else:
            self._counts[index] += 1
            self._total_count += 1

    def set_counts(self, counts):
        """fast path for BinaryTable: all counts at once"""
        self._counts = np.asarray(counts, dtype=np.int64)
        self._total_count = int(self._counts.sum())

    def _get_index(self, bitset):
        idx = 0
        d = self._dimensions
        for i in range(d):
            if bitset.get(i) if hasattr(bitset, "get") else bitset[i]:
                idx += int(math.pow(2, d - i - 1))
        return idx

    def get_entropy(self):
        """NOTE TODO (java) rewrite to reduce rounding errors"""
        if self._total_count == 0:
            return 0.0
        entropy = 0.0
        total = float(self._total_count)
        ln2 = math.log(2.0)
        for c in self._counts.tolist():
            frac = c / total
            if frac > 0.0:
                entropy += (-frac * math.log(frac) / ln2)
        return entropy

    # obsolete in java, kept
    def set_count(self, index, count):
        if hasattr(index, "get"):
            index = self._get_index(index)
        if index >= self._size:
            log.error("out of bounds")
        else:
            old = int(self._counts[index])
            self._counts[index] = count
            self._total_count += count - old

    def get_count(self, index):
        if hasattr(index, "get"):
            index = self._get_index(index)
        return int(self._counts[index])

    def get_exclusive_coverage(self):
        if self._total_count == 0:
            return 0
        cov = 0.0
        for i in range(self._dimensions):
            cov += int(self._counts[int(math.pow(2, i))])
        return cov

    def get_overlap(self):
        return int(self._counts[-1])

    def get_bit_set(self, index):
        from .item_set import ItemSet
        b = ItemSet(self._dimensions)
        i = index
        for k in range(self._dimensions - 1, -1, -1):
            if i >= int(math.pow(2, k)):
                b.set(self._dimensions - k - 1)
                i -= int(math.pow(2, k))
        return b

    def get_dimensions(self):
        return self._dimensions

    def get_total_count(self):
        return self._total_count

    def get_bdeu(self):
        """Riggelsen 2006 page 27, 3.15: Bayesian Dirichlet Equivalent Uniform"""
        if self._total_count == 0:
            return 0
        q_i = self.get_size() // 2
        a_ijk = 1 / float(self.get_size())
        a_ij = 1 / float(q_i)
        lg_ijk = function.log_gamma(a_ijk)
        lg_ij = function.log_gamma(a_ij)
        q = 0.0
        for j in range(q_i):
            s = 0.0
            post = 0.0
            c0 = self.get_count(j * 2)
            post += function.log_gamma(a_ijk + c0) - lg_ijk
            s += c0
            c1 = self.get_count(j * 2 + 1)
            post += function.log_gamma(a_ijk + c1) - lg_ijk
            s += c1
            q += lg_ij - function.log_gamma(a_ij + s) + post
        return q
