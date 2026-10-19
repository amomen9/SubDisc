"""2D kernel density estimate of two numeric columns, used by the result
window to draw the density of a subgroup (java ProbabilityDensityFunction2_2D,
with the Silverman bandwidth of ProbabilityDensityFunctionMV)."""

import math

import numpy as np

from .histo import vec

CUTOFF = 2.0
SAMPLES = 101


def h_silverman(data):
    """diagonal bandwidth matrix, sqrt(H_ii) = fac * std_dev_i"""
    d = len(data)
    n = float(len(data[0]))
    p = 1.0 / (d + 4.0)
    fac = math.pow(4.0 / (d + 2.0), p) * (math.pow(n, -p) if n > 0 else float('inf'))
    H = [[0.0] * d for _ in range(d)]
    for i in range(d):
        h_ii = fac * vec.std_dev(list(data[i]), True)
        H[i][i] = h_ii * h_ii
    return H


class ProbabilityDensityFunction2D:
    def __init__(self, columns, members):
        m = np.asarray(members, dtype=bool)
        self.data = [c.floats_view()[:c.size()][m[:c.size()]].astype(np.float64) for c in columns[:2]]
        x, y = self.data
        self._min_x = float(x.min()) if x.size else float('nan')
        self._max_x = float(x.max()) if x.size else float('nan')
        self._min_y = float(y.min()) if y.size else float('nan')
        self._max_y = float(y.max()) if y.size else float('nan')
        self.hs = h_silverman(self.data)
        self.std_devs = [vec.std_dev(list(d), False) for d in self.data]
        self._low_x = self._high_x = self._low_y = self._high_y = float('nan')
        self.grids = []
        for i in range(2):
            h = self.hs[i][i]
            n = self._grid_size(self.data[i], h, self.std_devs[i], i == 0)
            self.grids.append(self._grid(self.data[i], self.std_devs[i], n))
        self.density = _density(self.grids[0], self.grids[1], self.data, self.hs)

    def _grid_size(self, data, h, sd, first):
        mn = float(np.min(data))
        mx = float(np.max(data))
        g_min = mn - CUTOFF * sd
        g_max = mx + CUTOFF * sd
        rng = g_max - g_min
        s = rng / h if h else float('inf')
        if first:
            self._low_x, self._high_x = g_min, g_max
        else:
            self._low_y, self._high_y = g_min, g_max
        k = int((math.ceil(s) * SAMPLES) / (2.0 * CUTOFF)) if math.isfinite(s) else 128
        k = min(k, 128)   # AK: maximum on k, avoids overly large matrices
        print("%f %f %f %f %f %d " % (g_min, g_max, rng, h, s, k))
        return max(k, 1)

    @staticmethod
    def _grid(data, sd, n):
        mn = float(np.min(data))
        mx = float(np.max(data))
        g_min = mn - CUTOFF * sd
        g_max = mx + CUTOFF * sd
        dx = (g_max - g_min) / n
        g = g_min + np.arange(n) * dx
        g[-1] = g_max   # possible rounding error
        return g

    def get_size_x(self):
        return self.density.shape[0]

    def get_size_y(self):
        return self.density.shape[1]

    def get(self, x, y):
        return float(self.density[x, y])

    def get_min_x(self):
        return self._min_x

    def get_max_x(self):
        return self._max_x

    def get_min_y(self):
        return self._min_y

    def get_max_y(self):
        return self._max_y

    def get_low_x(self):
        return self._low_x

    def get_high_x(self):
        return self._high_x

    def get_low_y(self):
        return self._low_y

    def get_high_y(self):
        return self._high_y

    def get_max_density(self):
        m = float(np.max(self.density)) if self.density.size else 0.0
        return max(0.0, m)


def _density(x_grid, y_grid, data, hs):
    cm = np.array(hs, dtype=np.float64)
    cm_i = np.linalg.inv(cm)
    f = 1.0 / math.sqrt(math.pow(2.0 * math.pi, 2) * np.linalg.det(cm))
    xd, yd = data
    out = np.zeros((len(x_grid), len(y_grid)))
    for i, gx in enumerate(x_grid):
        dx = gx - xd
        dy = y_grid[:, None] - yd[None, :]
        q = (dx * dx * cm_i[0, 0])[None, :] + dx[None, :] * dy * (cm_i[0, 1] + cm_i[1, 0]) + dy * dy * cm_i[1, 1]
        out[i] = np.exp(-0.5 * q).sum(axis=1) * f
    return out
