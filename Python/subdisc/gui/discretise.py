"""The hidden 'Discretise' button (ADD_DISCRETISE_BUTTON = False in java).

Writes copies of the Table to csv files where every numeric column (except
the primary target) is binned with exactly the same code as the coarse
half-interval (.bNNNN) and interval (.nNNNN) strategies in
SubgroupDiscovery. Development code, kept for completeness.
"""

import bisect
import os

import numpy as np

from ..attribute_type import AttributeType
from ..column import Column
from ..interval import Interval
from ..subgroup_discovery import get_next_bin_boundary as _next
from ..table import Table
from ..java_compat import float_to_string

INT_MIN = -2147483648


def discretise_all(table, target_concept):
    for c in table.get_columns():
        if c.get_type() == AttributeType.NUMERIC:
            c.build_sorted(None)
    for b in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        _half_intervals(table, target_concept, b)
    # Vikamine default is 3 bins, SubDisc default is 8 bins
    # NOTE (java) with minimum support = 10%, more than 10 bins is useless
    for b in (2, 3, 4, 5, 6, 7, 8, 9, 10):
        _intervals(table, target_concept, b)
    print()


def _new_table(table, suffix):
    name = "%s.%s" % (table.get_source(), suffix)
    path = name + ".csv"
    t = Table(path, name, table.get_nr_rows(), table.get_nr_columns())
    return t, path


def _half_intervals(table, tc, nr_bins):
    print("\nBINS: " + str(nr_bins))
    n = table.get_nr_rows()
    bs = np.ones(n, dtype=bool)
    t, path = _new_table(table, "b%04d" % nr_bins)
    cs = t.get_columns()
    for c in table.get_columns():
        if c.get_type() != AttributeType.NUMERIC or c is tc.get_primary_target():
            cs.append(c)
            continue
        bounds = []
        min_cov = INT_MIN   # FAKE: never true
        b = 1
        counts = c.get_value_count(bs).itsCounts
        nxt = _next(n, b, nr_bins)
        cover = 0
        i = 0
        while b < nr_bins and i < len(counts):   # java has no bounds check here
            cnt = int(counts[i])
            if cnt != 0:
                cover += cnt
                if not (cover <= nxt or cover < min_cov):
                    if cover == n:
                        break
                    bounds.append(c.get_sorted_value(i))
                    b += 1
                    nxt = _next(n, b, nr_bins)
                    while nxt <= cover - 1:
                        b += 1
                        nxt = _next(n, b, nr_bins)
            i += 1
        # binary search needs all upper bounds, add the max when missing
        x = c.get_max()
        if not bounds or bounds[-1] != x:
            bounds.append(x)
        f = c.floats_view()[:c.size()]
        prev = 0
        for v in bounds:
            cnt = int(np.count_nonzero(f <= np.float32(v)))
            print("%s\t%s\t%d" % (c.get_name(), float_to_string(v), cnt - prev))
            prev = cnt
        nc = Column(c.get_name(), c.get_short(), c.get_type(), c.get_index(), c.size())
        for v in f.tolist():
            k = bisect.bisect_left(bounds, v)
            nc.add_float(bounds[min(k, len(bounds) - 1)])
        nc.close()
        cs.append(nc)
    t.to_file(None, path)
    print(os.path.abspath(path))


def _intervals(table, tc, nr_bins):
    print("\nBINS: " + str(nr_bins))
    n = table.get_nr_rows()
    bs = np.ones(n, dtype=bool)
    t, path = _new_table(table, "n%04d" % nr_bins)
    cs = t.get_columns()
    for c in table.get_columns():
        if c.get_type() != AttributeType.NUMERIC or c is tc.get_primary_target():
            cs.append(c)
            continue
        bounds = []
        min_cov = INT_MIN
        b = 1
        counts = c.get_value_count(bs).itsCounts
        last_cover = 0
        lo = float("-inf")
        nxt = _next(n, b, nr_bins)
        cover = 0
        i = 0
        while i < len(counts) and b < nr_bins:
            cnt = int(counts[i])
            if cnt != 0:
                cover += cnt
                if not (cover <= nxt or cover < min_cov):
                    if cover == n:
                        break
                    hi = c.get_sorted_value(i)
                    bounds.append(Interval(lo, hi))
                    last_cover = cover
                    lo = hi
                    b += 1
                    nxt = _next(n, b, nr_bins)
                    while nxt <= cover - 1:
                        b += 1
                        nxt = _next(n, b, nr_bins)
            i += 1
        if last_cover != 0 and last_cover != n and lo != float("inf"):
            bounds.append(Interval(lo, float("inf")))
        f = c.floats_view()[:c.size()]
        for iv in bounds:
            print("%s\t%s\t%d" % (c.get_name(), iv, sum(1 for v in f.tolist() if iv.between(v))))
        nc = Column(c.get_name(), c.get_short(), AttributeType.NOMINAL, c.get_index(), c.size())
        for v in f.tolist():
            for iv in bounds:
                if iv.between(v):
                    nc.add_nominal(str(iv).replace(", ", ";"))
        nc.close()
        cs.append(nc)
    t.to_file(None, path)
    print(os.path.abspath(path))
