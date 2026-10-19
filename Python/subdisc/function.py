"""Static helpers shared by a few classes (log gamma, distinct float values)."""

import math

import numpy as np

from .java_compat import java_sort_floats, float_compare

# for log_gamma
GAMMA_STP = 2.50662827465
GAMMA_C1 = 76.18009173
GAMMA_C2 = -86.50532033
GAMMA_C3 = 24.01409822
GAMMA_C4 = -1.231739516
GAMMA_C5 = 1.20858003e-3
GAMMA_C6 = -5.36382e-6
TABULATION = 100000

# itsPrecomputed[i] = sum(log(2..i-1)), same loop order as java so the sums
# are bit-identical
_precomputed = [0.0] * TABULATION
_v = 0.0
for _i in range(2, TABULATION):
    _precomputed[_i] = _v
    _v += math.log(_i)
del _v, _i


def log_gamma(x):
    """Used by BinaryTable.compute_bdeu_faster() and CrossCube.get_bdeu()."""
    ser = (1.0 + GAMMA_C1 / x + GAMMA_C2 / (x + 1.0) + GAMMA_C3 / (x + 2.0) + GAMMA_C4 / (x + 3.0)
           + GAMMA_C5 / (x + 4.0) + GAMMA_C6 / (x + 5.0))
    return (x - 0.5) * math.log(x + 4.5) - x - 4.5 + math.log(GAMMA_STP * ser)


def log_gamma_big(a):
    """Iyad Batal: log of gamma, quite slow for data larger than TABULATION."""
    if a < TABULATION:
        return _precomputed[a]
    return log_gamma_slow(a)


def log_gamma_slow(a):
    res = 0.0
    for i in range(2, a):
        res += math.log(i)
    return res


def get_cardinality(arr):
    """Number of distinct floats, Float.compare() style (NaN == NaN, -0.0 != 0.0)."""
    n = len(arr)
    if n <= 1:
        return n
    fa = java_sort_floats(arr)
    # nan's are together at the end, count them once
    nan_mask = np.isnan(fa)
    real = fa[~nan_mask]
    count = 1 if nan_mask.any() else 0
    if real.size:
        # zero sign matters for Float.compare
        diff = np.ones(real.size, dtype=bool)
        diff[1:] = (real[1:] != real[:-1]) | (np.signbit(real[1:]) != np.signbit(real[:-1]))
        count += int(np.count_nonzero(diff))
    return count


def get_unique_values(arr):
    """Sorted array with every distinct value once (Float.compare() equality)."""
    n = len(arr)
    if n <= 1:
        return np.array(arr, dtype=np.float32).copy()
    fa = java_sort_floats(arr)
    keep = np.ones(n, dtype=bool)
    a, b = fa[1:], fa[:-1]
    same = ((a == b) & (np.signbit(a) == np.signbit(b))) | (np.isnan(a) & np.isnan(b))
    keep[1:] = ~same
    return fa[keep]
