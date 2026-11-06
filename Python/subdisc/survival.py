"""Survival (time-to-event) targets: the log-rank test of a subgroup against
its complement, and Kaplan-Meier curves. Not in the java version.

The target is two columns: a numeric time and an event indicator (binary,
or numeric where 1 means the event happened and 0 means censored, i.e. the
row left the study before anything happened). Rows with a missing time or
event take no part at all.

Notation, per distinct time t_j at which at least one event happens:
    n_j   rows still at risk (time >= t_j), d_j events at t_j
    n_1j, d_1j   the same, but only counting the subgroup
    E_1 = sum_j d_j * n_1j / n_j                  expected events in the subgroup
    O_1 = sum_j d_1j                              observed events in the subgroup
    V   = sum_j d_j (n_1j/n_j)(1 - n_1j/n_j)(n_j - d_j)/(n_j - 1)
    z   = (O_1 - E_1) / sqrt(V),  chi2 = z^2 (1 degree of freedom)
z > 0: more events than expected, so the subgroup does worse than the rest.
"""

import math

import numpy as np

from .attribute_type import AttributeType
from .qm import QM
from .target_type import TargetType


def event_flags(column):
    """bool array of the event column: binary as is, numeric != 0"""
    n = column.size()
    if column.get_type() == AttributeType.BINARY:
        return np.asarray(column.get_binaries()[:n], dtype=bool)
    if column.get_type() == AttributeType.NUMERIC:
        f = np.asarray(column.floats_view()[:n], dtype=np.float64)
        return np.nan_to_num(f, nan=0.0) != 0.0
    raise ValueError("event column '%s' must be binary or numeric, not %s" % (column.get_name(), column.get_type()))


def valid_rows(time_column, event_column):
    """rows with a usable time and event"""
    n = min(time_column.size(), event_column.size())
    t = np.asarray(time_column.floats_view()[:n], dtype=np.float64)
    ok = ~np.isnan(t)
    ok &= ~np.asarray(time_column.missing_view()[:n], dtype=bool)
    ok &= ~np.asarray(event_column.missing_view()[:n], dtype=bool)
    return ok


def chi2_p_value(chi2):
    """upper tail of a chi-squared with 1 df: P(X >= chi2) = erfc(sqrt(chi2/2))"""
    if chi2 != chi2:
        return float('nan')
    if chi2 <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(chi2 / 2.0))


class SurvivalBase:
    """Everything about the whole (selected) data that does not depend on the
    subgroup, computed one time; evaluate() is then O(rows) per subgroup."""

    def __init__(self, time_column, event_column, selection=None):
        self.time_column = time_column
        self.event_column = event_column
        ok = valid_rows(time_column, event_column)
        if selection is not None:
            m = min(ok.shape[0], selection.shape[0])
            ok = ok[:m] & np.asarray(selection[:m], dtype=bool)
        self.nr_rows = ok.shape[0]
        self.rows = np.flatnonzero(ok)
        t = np.asarray(time_column.floats_view()[:self.nr_rows], dtype=np.float64)[self.rows]
        e = event_flags(event_column)[:self.nr_rows][self.rows]
        order = np.argsort(t, kind="stable")
        self.rows = self.rows[order]          # original row numbers, by time
        self.times = t[order]
        self.events = e[order]
        n = self.times.shape[0]
        self.size = n
        ev_times = np.unique(self.times[self.events])
        self.event_times = ev_times
        # [start, end) of every event time in the sorted arrays
        self.start = np.searchsorted(self.times, ev_times, side="left")
        self.end = np.searchsorted(self.times, ev_times, side="right")
        self.n_at_risk = (n - self.start).astype(np.float64)
        ev_cum = np.concatenate(([0], np.cumsum(self.events, dtype=np.int64)))
        self.d = (ev_cum[self.end] - ev_cum[self.start]).astype(np.float64)
        self.total_events = int(ev_cum[-1])
        # the variance factor (n_j - d_j)/(n_j - 1), 0 where n_j == 1
        with np.errstate(all="ignore"):
            self.var_factor = np.where(self.n_at_risk > 1.0,
                                       (self.n_at_risk - self.d) / (self.n_at_risk - 1.0), 0.0)

    def sorted_mask(self, members):
        """members (rows of the table) in the time order of this base"""
        m = np.asarray(members, dtype=bool)
        out = np.zeros(self.size, dtype=bool)
        inside = self.rows < m.shape[0]
        out[inside] = m[self.rows[inside]]
        return out

    def log_rank(self, members):
        """(z, chi2, observed, expected, hazard ratio) of members against the rest"""
        ms = self.sorted_mask(members)
        n1 = int(np.count_nonzero(ms))
        if n1 == 0 or n1 == self.size or self.event_times.size == 0:
            return 0.0, 0.0, 0, 0.0, float('nan')
        # subgroup rows at risk from index k on: a suffix sum
        suffix = np.concatenate((np.cumsum(ms[::-1], dtype=np.int64)[::-1], [0]))
        n1j = suffix[self.start].astype(np.float64)
        ev_in = np.concatenate(([0], np.cumsum(ms & self.events, dtype=np.int64)))
        d1j = (ev_in[self.end] - ev_in[self.start]).astype(np.float64)
        frac = n1j / self.n_at_risk
        expected = float(np.sum(self.d * frac))
        observed = float(np.sum(d1j))
        var = float(np.sum(self.d * frac * (1.0 - frac) * self.var_factor))
        if var <= 0.0:
            return 0.0, 0.0, int(observed), expected, float('nan')
        z = (observed - expected) / math.sqrt(var)
        # the usual log-rank estimate of the hazard ratio: (O1/E1) / (O0/E0)
        o0 = self.total_events - observed
        e0 = self.total_events - expected
        if expected > 0.0 and o0 > 0.0 and e0 > 0.0:
            hr = (observed / expected) / (o0 / e0)
        else:
            hr = float('inf') if observed > 0 else float('nan')
        return z, z * z, int(observed), expected, hr

    def kaplan_meier(self, members=None):
        """(times, survival) step points of the Kaplan-Meier curve, and the
        censored times (for the tick marks); members None means all rows"""
        ms = np.ones(self.size, dtype=bool) if members is None else self.sorted_mask(members)
        t = self.times[ms]
        e = self.events[ms]
        return kaplan_meier(t, e)

    def median(self, members=None):
        times, surv, _ = self.kaplan_meier(members)
        return km_median(times, surv)


def kaplan_meier(times, events):
    """times sorted ascending; S(t) = prod over event times t_j <= t of (1 - d_j/n_j)"""
    times = np.asarray(times, dtype=np.float64)
    events = np.asarray(events, dtype=bool)
    n = times.shape[0]
    if n == 0:
        return np.array([0.0]), np.array([1.0]), np.array([])
    uniq = np.unique(times[events])
    start = np.searchsorted(times, uniq, side="left")
    end = np.searchsorted(times, uniq, side="right")
    ev_cum = np.concatenate(([0], np.cumsum(events, dtype=np.int64)))
    d = (ev_cum[end] - ev_cum[start]).astype(np.float64)
    at_risk = (n - start).astype(np.float64)
    surv = np.cumprod(1.0 - d / at_risk)
    xs = np.concatenate(([0.0], uniq))
    ys = np.concatenate(([1.0], surv))
    censored = times[~events]
    return xs, ys, censored


def km_median(xs, ys):
    """first time the curve reaches 0.5 or lower, NaN when it never does"""
    below = np.flatnonzero(ys <= 0.5)
    if below.size == 0:
        return float('nan')
    return float(xs[below[0]])


def survival_at(xs, ys, t):
    """S(t) of a step curve from kaplan_meier()"""
    k = int(np.searchsorted(xs, t, side="right")) - 1
    return float(ys[max(k, 0)])


class SurvivalMeasure:
    """quality of a subgroup for one of the survival QMs"""

    def __init__(self, qm, time_column, event_column, selection=None):
        if qm not in QM.get_quality_measures(TargetType.SURVIVAL):
            raise ValueError("%s is not a survival quality measure" % qm)
        self.qm = qm
        self.base = SurvivalBase(time_column, event_column, selection)

    def evaluate(self, members):
        """(quality, hazard ratio, median time of the subgroup)"""
        z, chi2, _o, _e, hr = self.base.log_rank(members)
        if self.qm == QM.LOGRANK:
            q = chi2
        elif self.qm == QM.LOGRANK_WORSE:
            q = z
        elif self.qm == QM.LOGRANK_BETTER:
            q = -z
        else:
            raise AssertionError(self.qm)
        return q, hr, self.base.median(members)

    def p_value(self, members):
        _z, chi2, _o, _e, _hr = self.base.log_rank(members)
        return chi2_p_value(chi2)
