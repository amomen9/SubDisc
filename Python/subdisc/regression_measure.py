"""Simple linear regression of the secondary target on the primary target,
for DOUBLE_REGRESSION (difference in slope between subgroup and complement).

Products are float32 (float * float in java), sums are double, in row
order. Cook's distance uses numpy linear algebra instead of Jama.
"""

import numpy as np

from . import log
from .java_compat import div, jsqrt, double_to_string, DOUBLE_MIN_VALUE
from .qm import QM
from .target_type import TargetType


def _seq(values):
    if values.size == 0:
        return 0.0
    return float(np.cumsum(values, dtype=np.float64)[-1])


class RegressionMeasure:
    def __init__(self, qm_type, primary, secondary):
        """base model from two columns"""
        self._qm = qm_type
        n = primary.size()
        self._sample_size = n
        x = primary.floats_view()[:n].astype(np.float32)
        y = secondary.floats_view()[:n].astype(np.float32)
        self._x = x
        self._y = y
        self._set_sums(x, y)
        self._compl_x = None
        self._compl_y = None
        self._base = None
        self._error_sq_sum = 0.0
        self._compl_error_sq_sum = 0.0
        self._slope = 0.0
        self._intercept = 0.0
        self._bounds = [0, 0, 0, 0, 0]
        if qm_type in (QM.REGRESSION_SSD_COMPLEMENT, QM.REGRESSION_SSD_DATASET, QM.REGRESSION_FLATNESS, QM.REGRESSION_SSD_4):
            self._update_regression_function()
            self._update_error_terms()
        elif qm_type == QM.COOKS_DISTANCE:
            self._update_regression_function()
            self._init_cook()
        else:
            if qm_type in QM.get_quality_measures(TargetType.DOUBLE_REGRESSION):
                raise AssertionError(qm_type)
            raise ValueError("Invalid argument: " + str(qm_type))

    def _set_sums(self, x, y):
        with np.errstate(all='ignore'):
            self._x_sum = _seq(x.astype(np.float64))
            self._y_sum = _seq(y.astype(np.float64))
            # float * float -> float in java, then added to a double
            self._xy_sum = _seq((x * y).astype(np.float64))
            self._x_sq_sum = _seq((x * x).astype(np.float64))
            self._y_sq_sum = _seq((y * y).astype(np.float64))

    @staticmethod
    def from_base(base, members):
        """non-base model for the rows in members, the rest is the complement"""
        rm = RegressionMeasure.__new__(RegressionMeasure)
        rm._qm = base._qm
        rm._base = base
        n = base._sample_size
        m = members[:n]
        rm._x = base._x[m]
        rm._y = base._y[m]
        rm._compl_x = base._x[~m]
        rm._compl_y = base._y[~m]
        rm._sample_size = int(rm._x.shape[0])
        rm._set_sums(rm._x, rm._y)
        rm._error_sq_sum = 0.0
        rm._compl_error_sq_sum = 0.0
        rm._slope = 0.0
        rm._intercept = 0.0
        rm._bounds = [0, 0, 0, 0, 0]
        return rm

    def get_quality_measure(self):
        return self._qm

    def get_evaluation_measure_value(self):
        self._update_regression_function()
        self._update_error_terms()
        return self.get_value()

    def get_value(self):
        """TODO (java) turn this t-value into a p-value"""
        b = self._base
        cxs = b._x_sum - self._x_sum
        cys = b._y_sum - self._y_sum
        cxss = b._x_sq_sum - self._x_sq_sum
        cxys = b._xy_sum - self._xy_sum
        cn = float(b._sample_size - self._sample_size)
        n = self._sample_size

        num = self._error_term_variance(self._error_sq_sum, n)
        den = self._x_sq_sum - div(2 * self._x_sum * self._x_sum, n) + div(self._x_sum * self._x_sum, n)
        var = div(num, den)
        # degenerate candidate subgroup, quality 0
        if n == 0 or n == 2 or den == 0:
            return 0
        num = self._error_term_variance(self._compl_error_sq_sum, cn)
        den = cxss - div(2 * cxs * cxs, cn) + div(cxs * cxs, cn)
        cvar = div(num, den)
        if cn == 0 or cn == 2 or den == 0:
            return 0
        slope = _get_slope(self._x_sum, self._y_sum, self._x_sq_sum, self._xy_sum, n)
        cslope = _get_slope(cxs, cys, cxss, cxys, cn)
        if var + cvar == 0:
            return 0
        log.log_command_line("\n           slope: " + double_to_string(slope))
        log.log_command_line("complement slope: " + double_to_string(cslope))
        q = self._qm
        if q in (QM.REGRESSION_SSD_COMPLEMENT, QM.REGRESSION_SSD_4):   # SSD_4: TODO (java)
            return div(abs(cslope - slope), jsqrt(var + cvar))
        if q == QM.REGRESSION_SSD_DATASET:
            return div(abs(b.get_slope() - slope), jsqrt(var + cvar))
        if q == QM.REGRESSION_FLATNESS:
            return -abs(slope)
        return 0.0

    def _update_regression_function(self):
        n = self._sample_size
        x_mean = div(self._x_sum, n)
        y_mean = div(self._y_sum, n)
        self._slope = _get_slope(self._x_sum, self._y_sum, self._x_sq_sum, self._xy_sum, n)
        self._intercept = y_mean - self._slope * x_mean

    def _update_error_terms(self):
        with np.errstate(all='ignore'):
            e = self._y.astype(np.float64) - (self._slope * self._x.astype(np.float64) + self._intercept)
            self._error_sq_sum = _seq(e * e)
            if self._base is not None:
                e = self._compl_y.astype(np.float64) - (self._slope * self._compl_x.astype(np.float64) + self._intercept)
                self._compl_error_sq_sum = _seq(e * e)

    @staticmethod
    def _error_term_variance(sq_sum, n):
        return div(sq_sum, n - 2)

    def get_slope(self):
        return self._slope

    def get_intercept(self):
        return self._intercept

    def get_base_function_value(self, x):
        return x * self._slope + self._intercept

    # ------------------------------------------------------------------
    # Cook's distance (not reachable from the search in java either)
    # ------------------------------------------------------------------

    def _init_cook(self):
        n = self._sample_size
        X = np.column_stack([np.ones(n), self._x.astype(np.float64)])
        Y = self._y.astype(np.float64).reshape(-1, 1)
        self._data_matrix = np.column_stack([self._x.astype(np.float64), self._y.astype(np.float64)])
        xtx_inv = np.linalg.inv(X.T @ X)
        self._beta_hat = xtx_inv @ X.T @ Y
        self._hat_matrix = X @ xtx_inv @ X.T
        self._residual = (np.eye(n) - self._hat_matrix) @ Y
        self._p = float(self._beta_hat.shape[0])
        self._s_squared = float((self._residual.T @ self._residual)[0, 0]) / (float(n) - self._p)
        r = np.sort(self._residual[:, 0] ** 2)
        self._r_squared = np.cumsum(r[::-1])[::-1]
        t = np.sort(np.diag(self._hat_matrix).copy())
        self._t = np.cumsum(t[::-1])[::-1]
        h = np.diag(self._hat_matrix)
        self._svp = self._residual[:, 0] ** 2 / self._p * h / (1 - h)

    def calculate(self, subgroup):
        members = subgroup.get_members()
        size = subgroup.get_coverage()
        if size < 2:
            self._bounds[4] += 1
            return DOUBLE_MIN_VALUE
        t = self._t[size]
        r2 = self._r_squared[size]
        b7 = self._bound(t, r2)
        if b7 > DOUBLE_MIN_VALUE:
            log.log_command_line("                   Bound 7: " + double_to_string(b7))
            self._bounds[0] += 1
        n = self._sample_size
        m = members[:n]
        idx = np.flatnonzero(m)
        removed = np.flatnonzero(~m)
        sq_res = float(np.sum(self._residual[removed, 0] ** 2))
        b6 = self._bound(t, sq_res)
        if b6 > DOUBLE_MIN_VALUE:
            log.log_command_line("                   Bound 6: " + double_to_string(b6))
            self._bounds[1] += 1
        trace = float(np.trace(self._hat_matrix[np.ix_(removed, removed)]))
        b5 = self._bound(trace, r2)
        if b5 > DOUBLE_MIN_VALUE:
            log.log_command_line("                   Bound 5: " + double_to_string(b5))
            self._bounds[2] += 1
        b4 = self._bound(trace, sq_res)
        if b4 > DOUBLE_MIN_VALUE:
            log.log_command_line("                   Bound 4: " + double_to_string(b4))
            self._bounds[3] += 1
        svp = float(np.sum(self._svp[removed]))
        log.log_command_line("                   SVP est: " + double_to_string(svp))
        sub = self._data_matrix[idx]
        if np.linalg.matrix_rank(sub) < 2:
            self._bounds[4] += 1
            return DOUBLE_MIN_VALUE
        X = np.column_stack([np.ones(size), sub[:, 0]])
        Y = sub[:, 1].reshape(-1, 1)
        xtx_inv = np.linalg.inv(X.T @ X)
        beta = xtx_inv @ X.T @ Y
        hat = X @ xtx_inv @ X.T
        res = (np.eye(size) - hat) @ Y
        p = float(beta.shape[0])
        s2 = float((res.T @ res)[0, 0]) / (float(n) - p)
        parent = np.array([[self._intercept], [self._slope]])
        d = beta - parent
        q = float((d.T @ X.T @ X @ d)[0, 0]) / (p * s2)
        log.log_refinement(double_to_string(q) + "," + double_to_string(svp) + "," + str(size))
        return q

    def _bound(self, t, value):
        if t >= 1:
            return DOUBLE_MIN_VALUE
        return t / ((1 - t) * (1 - t)) * value / (self._p * self._s_squared)

    def get_nr_bound_seven(self):
        return self._bounds[0]

    def get_nr_bound_six(self):
        return self._bounds[1]

    def get_nr_bound_five(self):
        return self._bounds[2]

    def get_nr_bound_four(self):
        return self._bounds[3]

    def get_nr_rank_def(self):
        return self._bounds[4]


def _get_slope(x_sum, y_sum, x_sq_sum, xy_sum, n):
    x_mean = div(x_sum, n)
    y_mean = div(y_sum, n)
    num = xy_sum - x_mean * y_sum - y_mean * x_sum + n * x_mean * y_mean
    den = x_sq_sum - 2 * x_mean * x_sum + x_sum * x_mean
    return div(num, den)
