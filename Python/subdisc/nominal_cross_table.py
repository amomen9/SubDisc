from bisect import bisect_left
from functools import cmp_to_key

import numpy as np

from .java_compat import int32


class NominalCrossTable:
    """Positive / negative counts per nominal value inside a subgroup."""

    def __init__(self, column, members, target):
        domain = column.get_shared_domain() or tuple(column.distinct_values())
        nr_members = int(np.count_nonzero(members))
        counts = column.get_unique_nominal_domain_counts(members, nr_members)
        present = [domain[i] for i in range(len(counts) - 1) if counts[i] > 0]
        self._values = sorted(present)
        n = len(self._values)
        self._pos = [0] * n
        self._neg = [0] * n
        if n == 0:
            return
        # value code -> position in the sorted value list
        lut = np.full(len(column.distinct_values()), -1, dtype=np.int64)
        for k, v in enumerate(self._values):
            lut[column.distinct_values_map()[v]] = k
        size = column.size()
        codes = column.nominal_codes_view()[:size]
        m = members[:size]
        idx = lut[codes[m]]
        t = target[:size][m]
        pos = np.bincount(idx[t], minlength=n)
        neg = np.bincount(idx[~t], minlength=n)
        self._pos = [int(x) for x in pos[:n]]
        self._neg = [int(x) for x in neg[:n]]

    def get_value(self, index):
        return self._values[index]

    def get_positive_count(self, index):
        return self._pos[index]

    def get_negative_count(self, index):
        return self._neg[index]

    def size(self):
        return len(self._values)

    def get_sorted_domain_indices(self):
        """value indices sorted on p/n ratio (stable, like Collections.sort)"""
        pos, neg = self._pos, self._neg

        def cmp(i1, i2):
            # int arithmetic in java, overflow included
            return int32(int32(pos[i2] * neg[i1]) - int32(pos[i1] * neg[i2]))

        return sorted(range(len(self._values)), key=cmp_to_key(cmp))
