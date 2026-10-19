"""Columns of bits: the multi-label targets, or the subgroups of a result set
(for the pattern team)."""

import math

import numpy as np

from . import log
from . import function
from .cross_cube import CrossCube
from .item_set import ItemSet


class BinaryTable:
    def __init__(self, table=None, columns=None, *, _columns=None, _nr_records=None):
        """BinaryTable(table, binary_columns)"""
        if _columns is not None:
            self._columns = _columns
            self._nr_records = _nr_records
            return
        n = table.get_nr_rows()
        self._columns = [c.get_binaries()[:n] for c in columns]
        self._nr_records = n

    @staticmethod
    def from_subgroup_set(table, subgroups):
        """every subgroup becomes a binary column"""
        n = table.get_nr_rows()
        cols = [s.get_members()[:n] for s in subgroups]
        return BinaryTable(_columns=cols, _nr_records=n)

    def get_row(self, index):
        b = ItemSet(len(self._columns))
        for i, c in enumerate(self._columns):
            b.set(i, bool(c[index]))
        return b

    def select_columns(self, item_set):
        cols = [self._columns[i] for i in range(item_set.get_dimensions()) if item_set.get(i)]
        return BinaryTable(_columns=cols, _nr_records=self._nr_records)

    def select_rows(self, members):
        m = members[:self._nr_records]
        k = int(np.count_nonzero(m))
        cols = [c[m] for c in self._columns]
        return BinaryTable(_columns=cols, _nr_records=k)

    def _cube_counts(self):
        d = len(self._columns)
        size = int(math.pow(2, d))
        if self._nr_records == 0:
            return np.zeros(size, dtype=np.int64)
        idx = np.zeros(self._nr_records, dtype=np.int64)
        for j, c in enumerate(self._columns):
            idx += c.astype(np.int64) * int(math.pow(2, d - j - 1))
        return np.bincount(idx, minlength=size)

    def compute_bdeu_faster(self):
        d = len(self._columns)
        size = int(math.pow(2, d))
        counts = self._cube_counts().tolist()
        total = self._nr_records
        if total == 0:
            return 0
        q = 0.0
        q_i = size // 2
        a_ijk = 1.0 / float(size)
        a_ij = 1.0 / float(q_i)
        lg_ijk = function.log_gamma(a_ijk)
        lg_ij = function.log_gamma(a_ij)
        lg = function.log_gamma
        for j in range(q_i):
            s = 0.0
            post = 0.0
            c0 = counts[j * 2]
            post += lg(a_ijk + c0) - lg_ijk
            s += c0
            c1 = counts[j * 2 + 1]
            post += lg(a_ijk + c1) - lg_ijk
            s += c1
            q += lg_ij - lg(a_ij + s) + post
        return q

    def get_approximate_miki(self, k):
        count = 0
        best_set = ItemSet(self.get_nr_columns(), 0)
        max_entropy = 0.0
        log.log_command_line("finding approximate " + str(k) + "-itemsets")
        from .java_compat import double_to_string
        for i in range(1, k + 1):
            temp = best_set
            for j in range(self.get_nr_columns()):
                if not best_set.get(j):
                    count += 1
                    ext = best_set.get_extension(j)
                    cube = self.select_columns(ext)._count_cross_cube()
                    e = cube.get_entropy()
                    if max_entropy < e:
                        temp = ext
                        max_entropy = e
                        log.log_command_line("found a new maximum: " + str(ext) + ": " + double_to_string(max_entropy))
            # FIXME MM (java) could break here
            if best_set is temp:
                log.log_command_line("no change for i = " + str(i))
            best_set = temp
        best_set.set_joint_entropy(max_entropy)
        log.log_command_line("nr of column scans: " + str(count))
        return best_set

    def compute_correlation(self, k, l):
        """phi coefficient (Pearson for two binary variables)"""
        a = self._columns[k]
        b = self._columns[l]
        n = self._nr_records
        na = int(np.count_nonzero(a))
        nb = int(np.count_nonzero(b))
        nab = int(np.count_nonzero(a & b))
        from .java_compat import int64
        prod = int64(int64(int64(na * nb) * (n - na)) * (n - nb))
        if prod == 0:
            return 0.0
        # long / float in java: both sides become float
        num = np.float32(int64(n * nab - na * nb))
        den = np.float32(math.sqrt(prod) if prod > 0 else float('nan'))
        with np.errstate(all='ignore'):
            return float(num / den)

    def _count_cross_cube(self):
        cube = CrossCube(len(self._columns))
        cube.set_counts(self._cube_counts())
        return cube

    def print(self):
        for i in range(self._nr_records):
            log.log_command_line("".join("1" if c[i] else "0" for c in self._columns))

    def get_nr_records(self):
        return self._nr_records

    def get_nr_columns(self):
        return len(self._columns)

    def add_column(self, bits):
        self._columns.append(bits)

    def get_column(self, index):
        return self._columns[index]
