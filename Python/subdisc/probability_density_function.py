"""Density estimates of a numeric target, for the distribution based quality
measures (Hellinger, Kullback-Leibler, CWRAcc).

ProbabilityDensityFunction is the old 1000-bin histogram. The search uses
ProbabilityDensityFunction2 (USE_ProbabilityDensityFunction2 = True), a
kernel density estimate on a grid of at most 1001 points.
"""

import math

import numpy as np

from . import log
from .java_compat import float_to_string, to_f32, fdiv, double_to_string
from .histo import vec

USE_ProbabilityDensityFunction2 = True

F32 = np.float32
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _seq_f32_table(increment, max_count):
    """t[c] = increment added c times in float32, one after the other"""
    if max_count <= 0:
        return np.zeros(1, dtype=np.float32)
    t = np.zeros(max_count + 1, dtype=np.float32)
    t[1:] = np.cumsum(np.full(max_count, increment, dtype=np.float32), dtype=np.float32)
    return t


class ProbabilityDensityFunction:
    DEFAULT_NR_BINS = 1000
    DETAIL = 256
    CUTOFF = 4.0

    def __init__(self, data=None, members=None, _base=None):
        """PDF(column) for the whole data, PDF(pdf, members) for a subgroup."""
        self._nr_bins = ProbabilityDensityFunction.DEFAULT_NR_BINS
        if isinstance(data, ProbabilityDensityFunction):
            base = data
            self._data = base._data
            self._density = np.zeros(base._nr_bins, dtype=np.float32)
            self._min = base._min
            self._max = base._max
            self._bin_width = base._bin_width
            self._absolute_count = int(np.count_nonzero(members))
            log.log_command_line("AbsoluteCount = " + str(self._absolute_count))
            inc = fdiv(1.0, self._absolute_count)
            vals = self._data.floats_view()[:self._data.size()][members[:self._data.size()]]
            self._add_all(vals, inc)
            return
        self._data = data
        self._density = np.zeros(self._nr_bins, dtype=np.float32)
        # TODO (java) include outlier treatment
        self._min = data.get_min()
        self._max = data.get_max()
        self._bin_width = fdiv(to_f32(self._max - self._min), self._nr_bins)
        log.log_command_line("Min = " + float_to_string(self._min))
        log.log_command_line("Max = " + float_to_string(self._max))
        log.log_command_line("BinWidth = " + float_to_string(self._bin_width))
        self._absolute_count = data.size()
        log.log_command_line("AbsoluteCount = " + str(self._absolute_count))
        inc = fdiv(1.0, self._absolute_count)
        n = data.size()
        vals = data.floats_view()[:n][~data.missing_view()[:n]]
        self._add_all(vals, inc)

    def _bins_of(self, vals):
        with np.errstate(all='ignore'):
            q = (vals.astype(np.float32) - F32(self._min)) / F32(self._bin_width)
        idx = np.where(np.isfinite(q), q, 0.0)
        idx = np.trunc(idx).astype(np.int64)
        idx = np.where(vals == F32(self._max), self._nr_bins - 1, idx)
        # java would crash on an index of nr_bins (rounding), keep it in range
        return np.clip(idx, 0, self._nr_bins - 1)

    def _add_all(self, vals, inc):
        """itsDensity[bin] += increment for every value, in float32"""
        if vals.size == 0:
            return
        counts = np.bincount(self._bins_of(vals), minlength=self._nr_bins)
        table = _seq_f32_table(inc, int(counts.max()))
        self._density = table[counts].astype(np.float32)

    def get_density(self, index):
        return float(self._density[index])

    def densities(self):
        return self._density

    def get_middle(self, index):
        return to_f32(F32(self._min) + F32(F32(index + 0.5) * F32(self._bin_width)))

    def print(self):
        log.log_command_line("ProbabilityDensityFunction:\n")
        for i, d in enumerate(self._density):
            log.log_command_line("  " + str(i) + "\t" + float_to_string(d))
        log.log_command_line("")

    def size(self):
        return self._nr_bins

    def get_absolute_count(self):
        return self._absolute_count

    @staticmethod
    def get_gaussian_distribution(sigma):
        if sigma <= 0.0 or math.isinf(sigma) or math.isnan(sigma):
            raise ValueError("Invalid sigma: " + str(sigma))
        width = int(2.0 * ProbabilityDensityFunction.CUTOFF * sigma)
        kernel = [0.0] * width
        correction = 0.0
        half = width // 2
        variance = sigma * sigma
        double_variance = 2.0 * variance
        factor = 1.0 / math.sqrt(math.pi * double_variance)
        for i in range(width):
            x = float(i - half)
            v = factor * math.exp((-x * x) / double_variance)
            kernel[i] = v
            correction += v
        for i in range(width):
            kernel[i] /= correction
        print("##### theSigma = %f\tKERNEL SIZE = %d" % (sigma, len(kernel)))
        return kernel

    def smooth(self, sigma=None):
        """destructive: replaces the histogram by its smoothed version"""
        if sigma is None:
            sigma = fdiv(to_f32(self._max - self._min), ProbabilityDensityFunction.DETAIL)
        length = self._nr_bins
        kernel = ProbabilityDensityFunction.get_gaussian_distribution(sigma / self._bin_width)
        out = [F32(0.0)] * length
        dens = self._density
        width = len(kernel)
        half = width // 2
        for i in range(half):
            corr = 0.0
            k = half + i
            n = 0
            while k >= 0:
                corr += kernel[k]
                out[i] = F32(float(out[i]) + kernel[k] * float(dens[n]))
                k -= 1
                n += 1
            out[i] = F32(float(out[i]) / corr)
        i = length - 1
        for j in range(half):
            corr = 0.0
            k = half + j
            n = length - 1
            while k >= 0:
                corr += kernel[k]
                out[i] = F32(float(out[i]) + kernel[k] * float(dens[n]))
                k -= 1
                n -= 1
            out[i] = F32(float(out[i]) / corr)
            i -= 1
        for i in range(half, length - half):
            acc = float(out[i])
            n = i - half
            for k in range(width):
                acc = float(F32(acc + kernel[k] * float(dens[n])))
                n += 1
            out[i] = F32(acc)
        self._density = np.array(out, dtype=np.float32)
        return self._density


class ProbabilityDensityFunction2(ProbabilityDensityFunction):
    """Kernel density estimate (Gaussian kernel, Silverman bandwidth) on a grid."""

    CUTOFF = 4.0
    SAMPLES = 1001

    def __init__(self, data, selection_or_members=None):
        if isinstance(data, ProbabilityDensityFunction2):
            base = data
            members = selection_or_members
            ProbabilityDensityFunction.__init__(self, base, members)
            self._data = base._data
            self._min2 = base._min2
            self._max2 = base._max2
            self._h = base._h
            self._hi = base._hi
            self._lo = base._lo
            self._dx = base._dx
            self._density2 = base._kde(members, base._density2.shape[0])
            self._complement_members = members
            self._complement_density = None
            return

        ProbabilityDensityFunction.__init__(self, data)
        selection = selection_or_members
        self._data = data
        n = data.size()
        f = data.floats_view()[:n]
        keep = ~data.missing_view()[:n]
        if selection is not None:
            keep = keep & selection[:n]
        values = f[keep].astype(np.float64)
        size = values.shape[0]
        self._max2 = float('-inf')
        self._min2 = float('inf')
        if size:
            self._max2 = max(self._max2, float(values.max()))
            self._min2 = min(self._min2, float(values.min()))
        sorted_vals = sorted(float(v) for v in values)
        self._h = vec.h_silverman(sorted_vals)
        self._lo = self._min2 - (ProbabilityDensityFunction2.CUTOFF * self._h)
        self._hi = self._max2 + (ProbabilityDensityFunction2.CUTOFF * self._h)
        rng = self._hi - self._lo
        # avoid trouble with Silverman returning 0
        self._h = max(self._h, rng / 1000) if not math.isnan(self._h) else self._h
        s = rng / self._h
        k = int((math.ceil(s) * ProbabilityDensityFunction2.SAMPLES) / (2.0 * ProbabilityDensityFunction2.CUTOFF))
        k = min(k, 1000)   # k could grow unlimited, giving a negative array size
        self._dx = rng / k
        print("h=%f lo=%f hi=%f range=%f s=%f k=%d dx=%f" % (self._h, self._lo, self._hi, rng, s, k, self._dx))
        if selection is None:
            # sic: java uses the first 'size' rows here, not the non missing ones
            all_data = np.zeros(n, dtype=bool)
            all_data[:size] = True
            self._density2 = self._kde(all_data, k + 1)
        else:
            self._density2 = self._kde(selection, k + 1)
        self._complement_members = None
        self._complement_density = None

    def _kde(self, members, nr_points):
        n = self._data.size()
        vals = self._data.floats_view()[:n][members[:n]].astype(np.float64)
        d_h = vals / self._h
        grid = self._lo + (np.arange(nr_points, dtype=np.float64) * self._dx)
        mu_h = grid / self._h
        density = np.zeros(nr_points, dtype=np.float32)
        cutoff = ProbabilityDensityFunction2.CUTOFF
        # java: for each grid point, for each member: density[i] += phi(diff)
        # (float += double), same order per grid point here
        with np.errstate(all='ignore'):
            for v in d_h:
                diff = v - mu_h
                inside = np.abs(diff) < cutoff
                if not inside.any():
                    continue
                add = np.where(inside, np.exp(-diff * diff / 2.0) / _SQRT_2PI, 0.0)
                density = (density.astype(np.float64) + add).astype(np.float32)
            nh = d_h.shape[0] * self._h
            density = (density.astype(np.float64) / nh).astype(np.float32)
        return density

    def get_density(self, index):
        return float(self._density2[index])

    def densities(self):
        return self._density2

    def get_complement_density(self, index):
        # FIXME MM (java) HACK; computed when first asked for
        if self._complement_density is None:
            n = self._data.size()
            compl = ~self._complement_members[:n]
            self._complement_density = self._kde(compl, self._density2.shape[0])
        return float(self._complement_density[index])

    def get_middle(self, index):
        return to_f32(self._lo + float(F32(index + 0.5)) * self._dx)

    def size(self):
        return self._density2.shape[0]

    def smooth(self, sigma=None):
        # all original code calls smooth() once, 'equivalent' to the KDE density
        return self._density2
