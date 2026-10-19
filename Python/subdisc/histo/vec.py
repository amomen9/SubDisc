"""Vector helpers (java Vec). Sums run left to right like the java loops,
so the bandwidths come out the same."""

import math

import numpy as np

from ..java_compat import float_compare, div, jsqrt, jpow, binary_search

ILLEGAL = (float('nan'), float('-inf'), float('inf'))


def is_valid(data):
    if data is None:
        return False
    if not is_sorted(data):
        return False
    return not contains_sorted(data, *ILLEGAL)


def is_sorted(data):
    for i in range(len(data) - 1):
        if float_compare(data[i], data[i + 1]) > 0:
            return False
    return True


def contains(data, *keys):
    for d in data:
        for k in keys:
            if float_compare(d, k) == 0:
                return True
    return False


def contains_sorted(data, *keys):
    for k in keys:
        if binary_search(data, k) >= 0:
            return True
    return False


def sum_(data):
    s = 0.0
    for v in data:
        s += float(v)
    return s


def add(data, x):
    for i in range(len(data)):
        data[i] = data[i] + x
    return data


def subtract(data, x):
    for i in range(len(data)):
        data[i] = data[i] - x
    return data


def multiply(data, x):
    for i in range(len(data)):
        data[i] = data[i] * x
    return data


def divide(data, x):
    for i in range(len(data)):
        data[i] = div(data[i], x)
    return data


def power(data, x):
    for i in range(len(data)):
        data[i] = jpow(data[i], x)
    return data


def integral(x, y):
    if len(x) != len(y):
        raise ValueError("|x| != |y|")
    if len(x) == 1:
        return float('nan')
    s = abs(y[0] * abs(x[0]))   # FIXME MM (java)
    for i in range(1, len(x)):
        s += y[i] * abs(x[i] - x[i - 1])
    return s


def expected_value(x, y):
    if len(x) != len(y):
        raise ValueError("|x| != |y|")
    e = 0.0
    for a, b in zip(x, y):
        e += a * b
    return div(e, sum_(y))


def minimum(data):
    m = data[0]
    for v in data[1:]:
        if float_compare(v, m) < 0:
            m = v
    return m


def maximum(data):
    m = data[0]
    for v in data[1:]:
        if float_compare(v, m) > 0:
            m = v
    return m


def mean(data):
    return div(sum_(data), len(data))


def spd(data, p):
    """sum of power deviations"""
    avg = mean(data)
    s = 0.0
    for v in data:
        s += jpow(float(v) - avg, p)
    return s


def variance(data, sample):
    return div(spd(data, 2), len(data) - (1 if sample else 0))


def std_dev(data, sample):
    return jsqrt(variance(data, sample))


def iqr(data):
    """assumes sorted data"""
    n = len(data)
    mid = n // 2
    q1 = data[mid // 2]
    q3 = data[(mid + (1 if (n & 1) == 1 else 0)) + (mid // 2)]
    return q3 - q1


def median(data):
    n = len(data)
    mid = n // 2
    if (n & 1) == 1:
        return data[mid]
    return (data[mid - 1] + data[mid]) / 2.0


def skew_non_parametric(data, med=None, sd=None):
    if med is None:
        return div(mean(data) - median(data), std_dev(data, False))
    return div(data - med, sd)


def skew(data):
    return div(central_moment(data, 3, False), jpow(variance(data, False), 1.5))


def kurtosis(data):
    return div(central_moment(data, 4, False), jpow(variance(data, False), 2.0)) - 3.0


def central_moment(data, moment, sample):
    if moment == 0:
        return 1.0
    if moment == 1:
        return 0.0
    if moment == 2:
        return variance(data, sample)
    if moment in (3, 4):
        return div(spd(data, moment), len(data))
    raise ValueError("choose moment = {0,1,2,3,4}")


def standardised_moment(data, moment, sample):
    if moment == 1:
        return 0.0
    if moment == 2:
        return 1.0
    if moment == 3:
        return skew(data)
    if moment == 4:
        return kurtosis(data)
    raise ValueError("choose moment = {1,2,3,4}")


def nr_bins(mn, mx, h):
    return math.ceil(div(mx - mn, h))


def k_doane(data):
    n = len(data)
    g1 = skew(data)
    sigma_g1 = jsqrt(div(6.0 * (n - 2), (n + 1) * (n + 3)))
    return 1.0 + log2(n) + log2(1.0 + div(abs(g1), sigma_g1))


def k_freeman_diaconis(data):
    return nr_bins(minimum(data), maximum(data), h_freeman_diaconis(data))


def h_freeman_diaconis(data):
    return div(2.0 * iqr(data), len(data) ** (1.0 / 3.0))


def k_rice(n):
    if not isinstance(n, int):
        n = len(n)
    return math.ceil(2.0 * n ** (1.0 / 3.0))


def k_scott(data):
    return nr_bins(minimum(data), maximum(data), h_scott(data))


def h_scott(data):
    return div(3.5 * std_dev(data, False), len(data) ** (1.0 / 3.0))


def k_silverman(data):
    return nr_bins(minimum(data), maximum(data), h_silverman(data))


def h_silverman(data):
    """0.9 * min(sd_s, IQR/1.34) * n^-0.2 (prints sd and iqr like java)"""
    n = len(data)
    sd = std_dev(data, True)
    q = iqr(data)
    from ..java_compat import double_to_string
    print("std_dev=" + double_to_string(sd) + " iqr=" + double_to_string(q))
    return 0.9 * min_java(sd, q / 1.34) * jpow(n, -0.2)


def min_java(a, b):
    if math.isnan(a) or math.isnan(b):
        return float('nan')
    return a if a <= b else b


def h_silverman_legacy(n, sd_sample):
    return jpow((4.0 * jpow(sd_sample, 5.0)) / (3.0 * n), 0.2)


def k_square_root(n):
    return math.sqrt(n)


def k_sturges(n):
    return math.ceil(1.0 + log2(n))


def log2(x):
    from ..java_compat import jlog
    return jlog(x) / math.log(2.0)


def unique_sorted(data):
    return compact(sorted(data))


def compact(data):
    n = len(data)
    if n == 0:
        return []
    out = [data[0]]
    for v in data[1:]:
        c = float_compare(out[-1], v)
        if c < 0:
            out.append(v)
        elif c > 0:
            raise AssertionError("data[i] > data[j]")
    return out


def range_(start, interval, size):
    return [start + (i * interval) for i in range(size)]


def mirror(z):
    size = 2 * len(z) - 1
    out = [0.0] * size
    out[len(z) - 1:] = list(z)
    j = 0
    for i in range(len(z) - 1, 0, -1):
        out[j] = -z[i]
        j += 1
    return out


def get_counts(data, keys):
    out = [0.0] * len(keys)
    for v in data:
        out[binary_search(keys, v)] += 1.0
    return out


def get_bin_counts(data, bins):
    out = [0.0] * len(bins)
    for v in data:
        idx = binary_search(bins, v)
        out[~idx if idx < 0 else idx] += 1.0
    return out
