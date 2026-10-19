"""Correlation between two numeric targets, for DOUBLE_CORRELATION."""

import math

import numpy as np

from .java_compat import div, jsqrt, jlog, double_to_string
from .qm import QM
from .target_type import TargetType


def _seq_sum(values):
    """sum in double, one value after the other (like the java +=)"""
    if values.size == 0:
        return 0.0
    return float(np.cumsum(values, dtype=np.float64)[-1])


class CorrelationMeasure:
    def __init__(self, qm_type, primary=None, secondary=None):
        """CorrelationMeasure(qm, primary_column, secondary_column) is the base
        model, CorrelationMeasure.copy_of(base) starts an empty subgroup model."""
        if not _is_valid_type(qm_type):
            raise _qm_exception(qm_type)
        self._base = None
        self._type = qm_type
        self._sample_size = 0
        self._x_sum = 0.0
        self._y_sum = 0.0
        self._xy_sum = 0.0
        self._x_sq_sum = 0.0
        self._y_sq_sum = 0.0
        self._correlation = 0.0
        self._correlation_outdated = True
        self._complement_outdated = True
        self._complement_correlation = float('nan')
        if primary is not None:
            n = primary.size()
            # addObservation(y = primary, x = secondary) in java
            self.add_observations(primary.floats_view()[:n], secondary.floats_view()[:n])

    @staticmethod
    def copy_of(base):
        if base is None:
            raise ValueError("Implementation: theBase should not be null")
        cm = CorrelationMeasure(base._type)
        cm._base = base
        return cm

    def add_observation(self, y, x):
        """note the order: y first (the java parameter names say so too)"""
        self._sample_size += 1
        self._x_sum += x
        self._y_sum += y
        self._xy_sum += x * y
        self._x_sq_sum += x * x
        self._y_sq_sum += y * y
        self._correlation_outdated = True
        self._complement_outdated = True

    def add_observations(self, ys, xs):
        """many observations at once, same sums as calling add_observation in order"""
        y = np.asarray(ys, dtype=np.float64)
        x = np.asarray(xs, dtype=np.float64)
        if y.size == 0:
            return
        if self._sample_size == 0:
            self._x_sum = _seq_sum(x)
            self._y_sum = _seq_sum(y)
            self._xy_sum = _seq_sum(x * y)
            self._x_sq_sum = _seq_sum(x * x)
            self._y_sq_sum = _seq_sum(y * y)
            self._sample_size = int(y.size)
        else:
            for a, b in zip(y.tolist(), x.tolist()):
                self.add_observation(a, b)
            return
        self._correlation_outdated = True
        self._complement_outdated = True

    def get_correlation(self):
        if self._correlation_outdated:
            return self._compute_correlation()
        return self._correlation

    def _compute_correlation(self):
        n = self._sample_size
        self._correlation = div(n * self._xy_sum - self._x_sum * self._y_sum,
                                jsqrt((n * self._x_sq_sum - self._x_sum * self._x_sum) * (n * self._y_sq_sum - self._y_sum * self._y_sum)))
        self._correlation_outdated = False
        return self._correlation

    def compute_correlation_distance(self):
        size = self.get_sample_size()
        compl = self._base.get_sample_size() - size
        if size <= 2 or compl <= 2:
            return 0.0
        return abs(self.get_complement_correlation() - self.get_correlation())

    def get_sample_size(self):
        return self._sample_size

    def get_x_sum(self):
        return self._x_sum

    def get_y_sum(self):
        return self._y_sum

    def get_xy_sum(self):
        return self._xy_sum

    def get_x_squared_sum(self):
        return self._x_sq_sum

    def get_y_squared_sum(self):
        return self._y_sq_sum

    def compare_to(self, other):
        a = self.get_evaluation_measure_value()
        b = other.get_evaluation_measure_value()
        return -1 if a < b else (1 if a > b else 0)

    def get_evaluation_measure_value(self):
        r = self.get_correlation()
        t = self._type
        if t == QM.CORRELATION_R:
            return r
        if t == QM.CORRELATION_R_NEG:
            return -r
        if t == QM.CORRELATION_R_SQ:
            return r * r
        if t == QM.CORRELATION_R_NEG_SQ:
            return -(r * r)
        if t == QM.CORRELATION_DISTANCE:
            return self.compute_correlation_distance()
        if t == QM.CORRELATION_P:
            return self.get_p_value()
        if t == QM.CORRELATION_ENTROPY:
            return self.compute_entropy()
        if t == QM.ADAPTED_WRACC:
            return self.compute_adapted_wracc()
        if t == QM.COSTS_WRACC:
            return self.compute_costs_wracc()
        if t == QM.CWTPD:
            return self.compute_cwtpd()
        if t == QM.TMCC:
            return self.compute_tmcc()
        if t == QM.MCC:
            return self.compute_mcc()
        if t == QM.PDC:
            return self.compute_pdc()
        if t == QM.MVPDC:
            return self.compute_mvpdc()
        raise AssertionError(t)

    @staticmethod
    def compare_em_values(first, second):
        """deprecated, never used"""
        if math.isnan(second):
            return True
        if math.isnan(first):
            return False
        return first >= second

    def get_complement_correlation(self):
        if self._complement_outdated:
            return self._compute_complement_correlation()
        return self._complement_correlation

    def _compute_complement_correlation(self):
        b = self._base
        if b is None:
            return float('nan')
        n = float(b.get_sample_size() - self.get_sample_size())
        xs = b.get_x_sum() - self.get_x_sum()
        ys = b.get_y_sum() - self.get_y_sum()
        xys = b.get_xy_sum() - self.get_xy_sum()
        xss = b.get_x_squared_sum() - self.get_x_squared_sum()
        yss = b.get_y_squared_sum() - self.get_y_squared_sum()
        self._complement_correlation = div(n * xys - xs * ys, jsqrt((n * xss - xs * xs) * (n * yss - ys * ys)))
        self._complement_outdated = False
        return self._complement_correlation

    def get_p_value(self):
        from .normal_distribution import NormalDistribution
        size = self.get_sample_size()
        compl = self._base.get_sample_size() - size
        if size <= 2 or compl <= 2:
            return 0.0
        nd = NormalDistribution()
        compl_size = float(self._base.get_sample_size() - size)
        sg_dev = div(1.0, jsqrt(size - 3))
        compl_dev = div(1.0, jsqrt(compl_size - 3))
        z = div(_fisher(self.get_correlation()) - _fisher(self.get_complement_correlation()), sg_dev + compl_dev)
        erf = nd.calc_erf(z / math.sqrt(2.0))
        p = 0.5 * (1.0 + erf)
        return p if p > 0.5 else 1.0 - p

    def compute_entropy(self):
        corr = self.compute_correlation_distance()
        if corr == 0.0:
            return 0.0
        frac = div(self._sample_size, float(self._base.get_sample_size())) if self._base is not None else 1.0
        weight = -1.0 * frac * jlog(frac) / math.log(2)
        return weight * corr

    def compute_adapted_wracc(self):
        n = self._sample_size
        return div((div(self._x_sum, n) - div(self._y_sum, n)) * n, self._base._sample_size)

    def compute_costs_wracc(self):
        b = self._base
        n = self._sample_size
        return ((self._x_sum - div(n * b._x_sum, b._sample_size))
                * (div(self._xy_sum, self._x_sum) - div(b._y_sum - self._y_sum, b._sample_size - n)))

    def compute_cwtpd(self):
        b = self._base
        dev = self._y_sum - (1 / float(b._sample_size)) * float(self._sample_size) * b._y_sum
        print("deviation from expectation", end="")
        print(double_to_string(dev))
        return div(dev * self._xy_sum, self._y_sum)

    def compute_tmcc(self):
        return self._y_sum * (div(self._xy_sum, self._y_sum) - div(self._x_sum - self._xy_sum, self._sample_size - self._y_sum))

    def compute_mcc(self):
        return div(self._xy_sum, self._y_sum) - div(self._x_sum - self._xy_sum, self._sample_size - self._y_sum)

    def compute_pdc(self):
        b = self._base
        return div(self._xy_sum, b._x_sum) - div(div(b._xy_sum, b._x_sum) * self._x_sum, b._x_sum)

    def compute_mvpdc(self):
        b = self._base
        print("true positives:", end="")
        print(double_to_string(self._y_sum))
        print("size subgroup:", end="")
        print(self._sample_size)
        print("total positives in data:", end="")
        print(double_to_string(b._y_sum))
        print("mean costs in subgroup and target:", end="")
        print(double_to_string(div(self._xy_sum, self._y_sum)))
        print("expectation", end="")
        print(double_to_string((1 / float(b._sample_size)) * float(self._sample_size) * b._y_sum))
        return self._xy_sum - div(b._xy_sum * self._x_sum, b._x_sum)


def _fisher(r):
    """z' = 0.5 ln[(1+r)/(1-r)]"""
    return 0.5 * jlog(div(1 + r, 1 - r))


def _is_valid_type(t):
    return t in QM.get_quality_measures(TargetType.DOUBLE_CORRELATION)


def _qm_exception(t):
    return ValueError("CorrelationMeasure: invalid QM-type '%s'" % t)
