"""Propensity scores (Rob Konijn): the expected target value per row, from
Bayes rule or from a logistic regression on the 'knowledge' subgroups.

Only reachable through autorun xml files: the gui hides PROP_SCORE_WRACC
and PROP_SCORE_RATIO as 'not implemented'.

Two notes on equivalence:
- java sizes its arrays with BitSet.size(), i.e. the row count rounded up to
  a multiple of 64. The padding rows are kept here on purpose (they enter
  the logistic model and the sum, as in java).
- java uses weka's Logistic (ridge 0). Here a plain maximum likelihood fit
  with Newton-Raphson is used; it is the same model, but the optimiser is
  different, so the last digits may differ.
"""

import numpy as np

from . import log
from .java_compat import double_to_string

BAYES_RULE = "BayesRule"
LOGISTIC_REGRESSION = "LogisticRegression"


def bitset_size(n):
    """java BitSet(n).size(): n rounded up to whole 64 bit words"""
    return ((n + 63) // 64) * 64


def padded(mask):
    n = mask.shape[0]
    size = bitset_size(n)
    out = np.zeros(size, dtype=bool)
    out[:n] = mask
    return out


class PropensityScore:
    def __init__(self, subgroup, target, local_kn, global_kn, method):
        self._method = method
        self._target = padded(np.asarray(target, dtype=bool))
        self._local = local_kn
        self._global = global_kn
        self._subgroup = subgroup
        self._scores = np.zeros(self._target.shape[0], dtype=np.float64)
        if method == BAYES_RULE:
            self._bayes_rule()
        elif method == LOGISTIC_REGRESSION:
            self._logistic_regression()
        else:
            self._sum = float('nan')
            return
        s = 0.0
        for d in self._scores.tolist():
            s += d
        self._sum = s

    def _bayes_rule(self):
        rules = list(self._global.get_statistics_bayes_rule2())
        rules += [r for r in self._local.get_statistics_bayes_rule(self._subgroup.get_conditions()) if r not in rules]
        print("Size overlapping subgroups:")
        print(len(rules))
        size = self._target.shape[0]
        p_t = float(np.count_nonzero(self._target)) / size
        p_not_t = 1.0 - p_t
        if not rules:
            print("Knowledge variables are empty!!!")
            self._scores[:] = p_t
            return
        t1 = np.full(size, p_t)
        t2 = np.full(size, p_not_t)
        for r in rules:
            given_t = r.get_probabilities_data_px_given_t()
            # sic: java multiplies term 2 with P(X|T) as well
            not_given_t = r.get_probabilities_data_px_given_t()
            t1 = t1 * given_t[:size]
            t2 = t2 * not_given_t[:size]
        with np.errstate(all='ignore'):
            self._scores = t1 / (t1 + t2)

    def _logistic_regression(self):
        vars_ = list(self._global.get_bit_sets2())
        for b in self._local.get_bit_sets(self._subgroup.get_conditions()):
            if not any(b is v or np.array_equal(b, v) for v in vars_):
                vars_.append(b)
        size = self._target.shape[0]
        X = np.zeros((size, len(vars_) + 1))
        X[:, 0] = 1.0
        for j, v in enumerate(vars_):
            X[:, j + 1] = padded(np.asarray(v, dtype=bool))[:size]
        y = self._target.astype(np.float64)
        try:
            beta = _fit_logistic(X, y)
            log.log_command_line("Logistic Regression model created")
            with np.errstate(all='ignore'):
                self._scores = 1.0 / (1.0 + np.exp(-(X @ beta)))
        except Exception as e:
            log.log_command_line("logistic classification failed")
            log.log_command_line(str(e))
        log.log_command_line("propensity score filled")

    def get_propensity_score(self):
        return self._scores

    def get_propensity_score_sum(self):
        return self._sum


def _fit_logistic(X, y, max_iter=200, tol=1e-10):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        with np.errstate(all='ignore'):
            p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = p * (1 - p)
        grad = X.T @ (y - p)
        H = X.T @ (X * w[:, None])
        step = np.linalg.lstsq(H, grad, rcond=None)[0]
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return beta


class StatisticsBayesRule:
    """P(X|T) and P(X|not T) per row for one knowledge subgroup. Arrays use
    the java BitSet.size() (padded) length, see the note at the top."""

    def __init__(self, x, target):
        x = padded(np.asarray(x, dtype=bool))
        t = padded(np.asarray(target, dtype=bool))
        size = t.shape[0]
        self._x = x
        self._target = t
        self._p_t = float(np.count_nonzero(t)) / size
        print("PT:")
        print(double_to_string(self._p_t))
        print("size target vector")
        print(size)
        not_t = ~t
        not_x = ~x
        with np.errstate(all='ignore'):
            self._p_x_given_t = np.count_nonzero(x & t) / float(np.count_nonzero(t))
            self._p_not_x_given_t = 1 - self._p_x_given_t
            self._p_not_x_given_not_t = np.count_nonzero(not_x & not_t) / float(np.count_nonzero(not_t))
            self._p_x_given_not_t = 1 - self._p_not_x_given_not_t
        self._given_t = np.where(x, self._p_x_given_t, self._p_not_x_given_t)
        self._given_not_t = np.where(x, self._p_x_given_not_t, self._p_not_x_given_not_t)

    def get_probabilities_data_px_given_t(self):
        return self._given_t

    def get_probabilities_data_px_given_not_t(self):
        return self._given_not_t
