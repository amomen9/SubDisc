"""Multiple testing correction for a mining result (not in the java version).

A search evaluates thousands of candidate subgroups, so some of them look
'significant' by pure chance: at alpha = 0.05 one in twenty null
candidates passes. The corrections below adjust the p-values for the
number of hypotheses m:

    Bonferroni   p_adj = min(1, m p)
    Holm         sort p_(1) <= ... ; p_adj(i) = max_{j<=i} min(1, (m - j + 1) p_(j))
    BH           Benjamini-Hochberg, controls the false discovery rate (FDR,
                 the expected share of false ones among the reported ones):
                 q(i) = min_{j>=i} min(1, m p_(j) / j)
    BY           Benjamini-Yekutieli, BH times c(m) = sum_{k=1..m} 1/k, valid
                 under any dependence between the tests (subgroups overlap,
                 so their tests are not independent)
    layered      Webb's layered critical values: alpha is split evenly over
                 the search depths and within a depth over its candidates,
                 a subgroup at depth L passes when p <= alpha / (D * C_L)
                 (D = maximum depth, C_L = candidates evaluated at depth L)

m can be the number of subgroups shown (optimistic: the search looked at
far more) or the number of candidates evaluated (honest). When m is larger
than the number of p-values given, the ones not given are taken to be 1.

The p-values can come from the p-Value column (Gaussian or empirical, from
random qualities) or from an exact test on the counts of each subgroup:
    single nominal   hypergeometric (Fisher) test of the 2x2 table
    single numeric   Welch t-test, subgroup against its complement
    survival         log-rank test
"""

import math

import numpy as np

from .qm import QM
from .target_type import TargetType

METHODS = ("Bonferroni", "Holm", "Benjamini-Hochberg (FDR)", "Benjamini-Yekutieli (FDR, any dependence)",
           "Layered critical values (Webb)")


# ----------------------------------------------------------------------
# adjustments, p is a list (in any order), m >= len(p)
# ----------------------------------------------------------------------

def _check(p, m):
    p = np.asarray(p, dtype=np.float64)
    if m is None:
        m = p.shape[0]
    if m < p.shape[0]:
        raise ValueError("m (%d) can not be smaller than the number of p-values (%d)" % (m, p.shape[0]))
    return p, int(m)


def bonferroni(p, m=None):
    p, m = _check(p, m)
    return np.minimum(1.0, p * m)


def holm(p, m=None):
    p, m = _check(p, m)
    order = np.argsort(p, kind="stable")
    k = np.arange(1, p.shape[0] + 1)
    adj = np.minimum(1.0, (m - k + 1) * p[order])
    adj = np.maximum.accumulate(adj)
    out = np.empty_like(p)
    out[order] = adj
    return out


def benjamini_hochberg(p, m=None):
    p, m = _check(p, m)
    order = np.argsort(p, kind="stable")
    k = np.arange(1, p.shape[0] + 1)
    adj = np.minimum(1.0, m * p[order] / k)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = adj
    return out


def benjamini_yekutieli(p, m=None):
    p, m = _check(p, m)
    c = float(np.sum(1.0 / np.arange(1, m + 1)))
    return np.minimum(1.0, benjamini_hochberg(p, m) * c)


def layered_thresholds(depths, counts_per_depth, alpha, max_depth=None):
    """alpha_L per subgroup (by its depth); counts_per_depth {depth: C_L}"""
    if max_depth is None:
        max_depth = max(counts_per_depth) if counts_per_depth else max(depths)
    out = []
    for d in depths:
        c = counts_per_depth.get(d, 0)
        out.append(alpha / (max_depth * c) if c > 0 else float('nan'))
    return np.array(out, dtype=np.float64)


def adjust(method, p, m, alpha=0.05, depths=None, counts_per_depth=None, max_depth=None):
    """(adjusted p-values or None, significant flags, per subgroup threshold or None)"""
    p = np.asarray(p, dtype=np.float64)
    if method == METHODS[4]:
        thr = layered_thresholds(depths, counts_per_depth or {}, alpha, max_depth)
        return None, p <= thr, thr
    if method == METHODS[0]:
        adj = bonferroni(p, m)
    elif method == METHODS[1]:
        adj = holm(p, m)
    elif method == METHODS[2]:
        adj = benjamini_hochberg(p, m)
    elif method == METHODS[3]:
        adj = benjamini_yekutieli(p, m)
    else:
        raise ValueError(method)
    return adj, adj <= alpha, None


# ----------------------------------------------------------------------
# exact tests from the counts
# ----------------------------------------------------------------------

def _log_choose(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeometric_p(N, K, n, x_obs, alternative="greater"):
    """N rows, K positives, a subgroup of n rows with x_obs positives.
    greater: P(X >= x_obs); less: P(X <= x_obs); two-sided: the total
    probability of all tables that are not more likely than the observed
    one (Fisher's exact test)."""
    lo = max(0, n + K - N)
    hi = min(n, K)
    if hi < lo:
        return 1.0
    x_obs = min(max(int(x_obs), lo), hi)
    xs = np.arange(lo, hi + 1, dtype=np.float64)
    # log pmf from lo on, by pmf(x+1)/pmf(x) = (K-x)(n-x) / ((x+1)(N-K-n+x+1))
    logp = np.empty(xs.shape[0])
    logp[0] = _log_choose(K, lo) + _log_choose(N - K, n - lo) - _log_choose(N, n)
    if xs.shape[0] > 1:
        x = xs[:-1]
        steps = np.log((K - x) * (n - x)) - np.log((x + 1) * (N - K - n + x + 1))
        logp[1:] = logp[0] + np.cumsum(steps)
    pmf = np.exp(logp - logp.max())
    pmf /= pmf.sum()
    i = x_obs - lo
    if alternative == "greater":
        p = pmf[i:].sum()
    elif alternative == "less":
        p = pmf[:i + 1].sum()
    else:
        p = pmf[pmf <= pmf[i] * (1 + 1e-7)].sum()
    return float(min(1.0, max(0.0, p)))


def _betacf(a, b, x):
    """continued fraction of the incomplete beta function (modified Lentz)"""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = tiny if abs(d) < tiny else d
        c = 1.0 + aa / c
        c = tiny if abs(c) < tiny else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def regularized_beta(a, b, x):
    """I_x(a, b)"""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def student_t_sf(t, df):
    """P(T >= t) for Student's t with df degrees of freedom"""
    if df <= 0 or t != t:
        return float('nan')
    x = df / (df + t * t)
    tail = 0.5 * regularized_beta(df / 2.0, 0.5, x)
    return tail if t >= 0 else 1.0 - tail


def welch_t(n1, mean1, var1, n0, mean0, var0):
    """(t, df) of Welch's test, var are the sample variances (n - 1)"""
    if n1 < 2 or n0 < 2:
        return float('nan'), float('nan')
    a, b = var1 / n1, var0 / n0
    if a + b <= 0:
        return float('nan'), float('nan')
    t = (mean1 - mean0) / math.sqrt(a + b)
    df = (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n0 - 1)) if (a > 0 or b > 0) else float('nan')
    return t, df


def p_from_t(t, df, alternative):
    if t != t or df != df:
        return float('nan')
    if alternative == "greater":
        return student_t_sf(t, df)
    if alternative == "less":
        return student_t_sf(-t, df)
    return min(1.0, 2.0 * student_t_sf(abs(t), df))


def normal_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


# which way a quality measure looks, for the one sided tests
_TWO_SIDED = {QM.ABSWRACC, QM.CHI_SQUARED, QM.INFORMATION_GAIN, QM.MUTUAL_INFORMATION, QM.PURITY,
              QM.ABS_Z_SCORE, QM.ABS_DEVIATION, QM.ABS_MEAN_TEST, QM.ABS_T_TEST, QM.EXPLAINED_VARIANCE,
              QM.SQUARED_HELLINGER, QM.SQUARED_HELLINGER_WEIGHTED, QM.SQUARED_HELLINGER_WEIGHTED_ADJUSTED,
              QM.KULLBACK_LEIBLER, QM.KULLBACK_LEIBLER_WEIGHTED, QM.CWRACC, QM.CWRACC_UNWEIGHTED, QM.LOGRANK}
_LESS = {QM.SPECIFICITY, QM.INVERSE_Z_SCORE, QM.INVERSE_AVERAGE, QM.INVERSE_SUM, QM.INVERSE_MEAN_TEST,
         QM.INVERSE_T_TEST, QM.LOGRANK_BETTER}


def alternative_for(qm):
    if qm in _TWO_SIDED:
        return "two-sided"
    if qm in _LESS:
        return "less"
    return "greater"


def has_exact_test(target_type):
    return target_type in (TargetType.SINGLE_NOMINAL, TargetType.SINGLE_NUMERIC, TargetType.SURVIVAL)


def full_mask(members, n):
    """members as a bool array of exactly n rows"""
    m = np.zeros(n, dtype=bool)
    mm = np.asarray(members, dtype=bool)[:n]
    m[:mm.shape[0]] = mm
    return m


def exact_p_values(subgroups, search_parameters, table, selection=None):
    """p-value of the exact test per subgroup (see the module doc)"""
    tt = search_parameters.get_target_type()
    tc = search_parameters.get_target_concept()
    qm = search_parameters.get_quality_measure()
    alt = alternative_for(qm)
    n_rows = table.get_nr_rows()
    base = np.ones(n_rows, dtype=bool) if selection is None else np.asarray(selection[:n_rows], dtype=bool)
    out = []
    if tt == TargetType.SINGLE_NOMINAL:
        from .condition import Condition
        from .condition_base import ConditionBase
        from .operators import Operator
        from .attribute_type import AttributeType
        target = tc.get_primary_target()
        cb = ConditionBase(target, Operator.EQUALS)
        value = tc.get_target_value()
        if target.get_type() == AttributeType.BINARY:
            cond = Condition(cb, AttributeType.is_valid_binary_true_value(value))
        else:
            cond = Condition(cb, value)
        y = target.evaluate(np.ones(n_rows, dtype=bool), cond)[:n_rows] & base
        N = int(np.count_nonzero(base))
        K = int(np.count_nonzero(y))
        for s in subgroups:
            m = full_mask(s.get_members(), n_rows) & base
            n = int(np.count_nonzero(m))
            x = int(np.count_nonzero(m & y))
            out.append(hypergeometric_p(N, K, n, x, alt))
    elif tt == TargetType.SINGLE_NUMERIC:
        target = tc.get_primary_target()
        f = np.asarray(target.floats_view()[:n_rows], dtype=np.float64)
        ok = base & ~np.isnan(f) & ~np.asarray(target.missing_view()[:n_rows], dtype=bool)
        for s in subgroups:
            m = full_mask(s.get_members(), n_rows)
            a, b = f[m & ok], f[~m & ok]
            if a.shape[0] < 2 or b.shape[0] < 2:
                out.append(float('nan'))
                continue
            t, df = welch_t(a.shape[0], a.mean(), a.var(ddof=1), b.shape[0], b.mean(), b.var(ddof=1))
            out.append(p_from_t(t, df, alt))
    elif tt == TargetType.SURVIVAL:
        from .survival import SurvivalBase, chi2_p_value
        sb = SurvivalBase(tc.get_primary_target(), tc.get_secondary_target(), selection)
        for s in subgroups:
            z, chi2, _o, _e, _hr = sb.log_rank(s.get_members())
            if alt == "two-sided":
                out.append(chi2_p_value(chi2))
            elif alt == "less":
                out.append(normal_sf(-z))
            else:
                out.append(normal_sf(z))
    else:
        raise ValueError("no exact test for target type '%s'" % tt)
    return out
