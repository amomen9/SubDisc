"""Probability mass function over equal width / equal height bins.

Only used by the TEMPORARY_CODE experiments in Process / SubgroupDiscovery
(switched off by default). XXX MM (java) extends the PDF only to keep that
temporary code simple.
"""

import math

import numpy as np

from . import column as column_module
from .attribute_type import AttributeType
from .column import Column
from .java_compat import binary_search, to_f32
from .operators import Operator
from .probability_density_function import ProbabilityDensityFunction


class ProbabilityMassFunctionND(ProbabilityDensityFunction):
    def __init__(self, column_or_parent, nr_split_points_or_members, use_equal_width=False):
        if isinstance(column_or_parent, ProbabilityMassFunctionND):
            parent = column_or_parent
            members = nr_split_points_or_members
            fake = Column("FAKE", "", AttributeType.NUMERIC, 0, 0)
            fake.close()
            self._data = fake
            self._density = np.zeros(self.DEFAULT_NR_BINS, dtype=np.float32)
            self._nr_bins = self.DEFAULT_NR_BINS
            self._min = self._max = self._bin_width = float('nan')
            self._absolute_count = 0
            self._bounds = None
            self._bin_indexes = None
            self.score = _qm(members, parent._bounds, parent._bin_indexes)
            return
        col = column_or_parent
        nr_split_points = nr_split_points_or_members
        ProbabilityDensityFunction.__init__(self, col)
        values = col.get_floats()
        mn = col.get_min()
        mx = col.get_max()
        if use_equal_width:
            self._bounds = _ew(nr_split_points, mn, mx)
        else:
            self._bounds = _eh(nr_split_points, col)
        self._bin_indexes = _bin(self._bounds, values)
        self.score = float('nan')
        _histo(self._bounds, self._bin_indexes)

    def get_pmf(self, members):
        size = len(self._bounds)
        idx = self._bin_indexes[np.asarray(members[:len(self._bin_indexes)], dtype=bool)]
        counts = np.bincount(idx, minlength=size)
        c = float(idx.shape[0])
        return counts / c


def _ew(nr_split_points, mn, mx):
    F = np.float32
    bounds = np.zeros(nr_split_points + 1, dtype=np.float32)
    width = F(F(F(mx) - F(mn)) / F(nr_split_points + 1))
    for i in range(nr_split_points):
        bounds[i] = F(F(mn) + F(F(i + 1) * width))
    bounds[-1] = mx
    return bounds


def _eh(nr_split_points, col):
    members = np.ones(col.size(), dtype=bool)
    orig = column_module.USE_NEW_BINNING
    column_module.USE_NEW_BINNING = True
    bounds = col.get_unique_split_points(members, nr_split_points, Operator.LESS_THAN_OR_EQUAL)
    column_module.USE_NEW_BINNING = orig
    if len(bounds) == 0:
        raise AssertionError("FIX eh()")
    mx = col.get_max()
    if float(bounds[-1]) != mx:
        bounds = np.append(bounds, np.float32(mx)).astype(np.float32)
    return bounds


def _bin(split_points, data):
    out = np.zeros(len(data), dtype=np.int64)
    for i, v in enumerate(data):
        idx = binary_search(split_points, v)
        out[i] = ~idx if idx < 0 else idx
    return out


def _qm(members, bounds, bin_indexes):
    n_bounds = len(bounds)
    m = np.asarray(members[:len(bin_indexes)], dtype=bool)
    cs = np.bincount(bin_indexes[m], minlength=n_bounds)
    cc = np.bincount(bin_indexes[~m], minlength=n_bounds)
    sg = float(np.count_nonzero(m))
    co = float(len(bin_indexes)) - sg
    print("#BOUND(<=)\tSUBGROUP\tCOMPLEMENT")
    for i in range(n_bounds):
        print("%f\t%f\t%f" % (bounds[i], cs[i] / sg, cc[i] / co))
    print()
    sum_p = 0.0
    sum_q = 0.0
    total = 0.0
    for i in range(n_bounds):
        p = cs[i] / sg
        q = cc[i] / co
        d = math.sqrt(p) - math.sqrt(q)
        total += d * d
        sum_p += p
        sum_q += q
    print("#SUMS:\tsumPi=%f\tsumQi=%f" % (sum_p, sum_q))
    return (1.0 / math.sqrt(2.0)) * math.sqrt(total)


def _histo(bounds, bin_indexes):
    counts = np.bincount(bin_indexes, minlength=len(bounds))
    print("\nPMF histogram:")
    print("BOUND (<=)\tCOUNT\tRELATIVE")
    n = to_f32(len(bin_indexes))
    for i in range(len(bounds)):
        print("%f\t%d\t%f" % (bounds[i], counts[i], counts[i] / n))
