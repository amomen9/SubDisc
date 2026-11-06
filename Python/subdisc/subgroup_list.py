"""MDL subgroup list (not in the java version).

A mining run gives the top-k subgroups by quality, and these overlap a lot
(on adult.txt every 'relationship = Husband' row is also a
'marital-status = Married-civ-spouse' row). A subgroup list is an ordered
list S_1, ..., S_k plus a default rule: a row belongs to the first
subgroup that covers it, rows that none covers go to the default rule.

The list is picked greedily with the Minimum Description Length (MDL)
principle: a subgroup only goes in when describing it (model bits) costs
less than what it saves on encoding the target of the rows it takes over
(data bits). The default rule always uses the distribution of the whole
data, so a subgroup has to stand out from the data, not from the rest of
the list (the idea of the SSD++ algorithm of Proenca et al.).

Notation, for a candidate S and the rows R it would take over (covered by
S and by no earlier subgroup, inside the mining selection):
    n        = |R|
    binary   p_0 share of positives in the data, p_S the share in R
             saved(S) = n * KL(p_S || p_0) - 1/2 log2(n)
    numeric  mu_0, sigma_0 of the data, mu_S, sigma_S of R
             saved(S) = n/2 log2(sigma_0^2 / sigma_S^2)
                        + (sum (y - mu_0)^2 / sigma_0^2 - n) / (2 ln 2) - log2(n)
    L(S)     = log* of the number of conditions + per condition
               log2(#attributes) + bits for its value (see model_bits())
    gain(S)  = saved(S) - L(S)       (normalised: gain(S) / n)
The 1/2 log2(n) per estimated parameter is the usual price of a parameter
that has to be sent along with the data.
"""

import math

import numpy as np

from .attribute_type import AttributeType
from .operators import Operator

LOG2_C0 = math.log2(2.865064)


def log_star(n):
    """Rissanen's universal code length for a positive integer, in bits"""
    if n < 1:
        raise ValueError(n)
    bits = LOG2_C0
    x = math.log2(n)
    while x > 0:
        bits += x
        x = math.log2(x) if x > 1 else 0.0
    return bits


def kl_binary(p, q):
    """KL(p || q) in bits for two Bernoulli distributions"""
    out = 0.0
    if p > 0:
        out += p * math.log2(p / q)
    if p < 1:
        out += (1 - p) * math.log2((1 - p) / (1 - q))
    return out


class ListEntry:
    __slots__ = ("subgroup", "rows", "new_rows", "statistic", "spread", "saved_bits", "model_bits", "gain")

    def describe_statistic(self, kind):
        if kind == "binary":
            return "p = %.4f" % self.statistic
        return "mu = %.4g, sigma = %.4g" % (self.statistic, self.spread)


class SubgroupList:
    """build with from_subgroups(), then predict() / score() on new rows"""

    def __init__(self, kind, nr_rows, default_statistic, default_spread=0.0):
        self.kind = kind                      # 'binary' or 'numeric'
        self.nr_rows = nr_rows                # rows the list was built on
        self.default_statistic = default_statistic
        self.default_spread = default_spread
        self.entries = []
        self.default_rows = 0
        self.default_only_bits = 0.0          # target encoded with the default rule alone
        self.total_gain = 0.0

    # ------------------------------------------------------------------
    # building
    # ------------------------------------------------------------------

    @staticmethod
    def from_subgroups(subgroups, kind, y, data_rows, table, nr_bins=8, normalised=False,
                       max_length=50, min_rows=1):
        """subgroups: candidates (any order); y: target per table row (bool for
        binary, float for numeric); data_rows: rows the mining used"""
        data_rows = np.asarray(data_rows, dtype=bool)
        if kind == "numeric":
            y = np.asarray(y, dtype=np.float64)
            data_rows = data_rows & ~np.isnan(y[:data_rows.shape[0]])
        else:
            y = np.asarray(y, dtype=bool)
        n_all = int(np.count_nonzero(data_rows))
        if n_all == 0:
            raise ValueError("no rows to build a subgroup list on")
        yd = y[:data_rows.shape[0]][data_rows]
        if kind == "binary":
            p0 = float(np.count_nonzero(yd)) / n_all
            sl = SubgroupList(kind, n_all, p0)
            sl.default_only_bits = n_all * _entropy(p0)
        else:
            mu0 = float(yd.mean())
            sd0 = float(yd.std())
            if sd0 <= 0.0:
                raise ValueError("the target is constant, nothing to describe")
            sl = SubgroupList(kind, n_all, mu0, sd0)
            sl.default_only_bits = _gauss_bits(yd, mu0, sd0)
        nr_attributes = max(1, sum(1 for c in table.get_columns() if c.get_is_enabled()) - 1)

        candidates = []
        for s in subgroups:
            m = np.asarray(s.get_members(), dtype=bool)
            m = m[:data_rows.shape[0]] & data_rows[:m.shape[0]]
            if m.shape[0] < data_rows.shape[0]:
                m = np.concatenate((m, np.zeros(data_rows.shape[0] - m.shape[0], dtype=bool)))
            candidates.append((s, m, model_bits(s.get_conditions(), nr_attributes, nr_bins)))

        free = data_rows.copy()
        used = set()
        while len(sl.entries) < max_length:
            best = None
            for k, (s, m, mbits) in enumerate(candidates):
                if k in used:
                    continue
                rows = m & free
                n = int(np.count_nonzero(rows))
                if n < max(1, min_rows):
                    continue
                saved, stat, spread = sl._saved_bits(y, rows, n)
                gain = saved - mbits
                score = gain / n if normalised else gain
                if gain > 0 and (best is None or score > best[0]):
                    best = (score, k, rows, n, saved, mbits, gain, stat, spread)
            if best is None:
                break
            _score, k, rows, n, saved, mbits, gain, stat, spread = best
            e = ListEntry()
            e.subgroup = candidates[k][0]
            e.rows = int(np.count_nonzero(candidates[k][1]))
            e.new_rows = n
            e.statistic = stat
            e.spread = spread
            e.saved_bits = saved
            e.model_bits = mbits
            e.gain = gain
            sl.entries.append(e)
            sl.total_gain += gain
            used.add(k)
            free &= ~rows
        sl.default_rows = int(np.count_nonzero(free))
        return sl

    def _saved_bits(self, y, rows, n):
        yr = y[:rows.shape[0]][rows]
        if self.kind == "binary":
            p = float(np.count_nonzero(yr)) / n
            return n * kl_binary(p, self.default_statistic) - 0.5 * math.log2(n), p, 0.0
        mu = float(yr.mean())
        sd = max(float(yr.std()), self.default_spread * 1e-3)
        sd0 = self.default_spread
        d0 = float(np.sum((yr - self.default_statistic) ** 2)) / (sd0 * sd0)
        saved = 0.5 * n * math.log2((sd0 * sd0) / (sd * sd)) + (d0 - n) / (2.0 * math.log(2.0)) - math.log2(n)
        return saved, mu, sd

    # ------------------------------------------------------------------
    # using it
    # ------------------------------------------------------------------

    def compression(self):
        """bits saved by the list relative to the default rule alone (0..1)"""
        if self.default_only_bits <= 0:
            return 0.0
        return self.total_gain / self.default_only_bits

    def assign(self, table, rows=None, evaluate=None):
        """index of the entry each row falls in, -1 for the default rule"""
        from .holdout import evaluate_conditions
        evaluate = evaluate or evaluate_conditions
        n = table.get_nr_rows()
        base = np.ones(n, dtype=bool) if rows is None else np.asarray(rows[:n], dtype=bool)
        out = np.full(n, -1, dtype=np.int64)
        free = base.copy()
        for k, e in enumerate(self.entries):
            m = evaluate(table, e.subgroup.get_conditions(), free)
            out[m] = k
            free &= ~m
        out[~base] = -2   # not asked for
        return out

    def predict(self, table, rows=None):
        """(prediction per row, spread per row) for the asked rows (NaN elsewhere)"""
        idx = self.assign(table, rows)
        stat = np.array([e.statistic for e in self.entries] + [self.default_statistic], dtype=np.float64)
        spread = np.array([e.spread for e in self.entries] + [self.default_spread], dtype=np.float64)
        pred = np.full(idx.shape[0], np.nan)
        sp = np.full(idx.shape[0], np.nan)
        asked = idx >= -1
        pick = np.where(idx[asked] == -1, len(self.entries), idx[asked])
        pred[asked] = stat[pick]
        sp[asked] = spread[pick]
        return pred, sp, idx

    def score(self, table, y, rows=None):
        """how well the list predicts the target on rows of table, with the
        parameters learnt on the training rows (nothing is refitted)"""
        pred, spread, idx = self.predict(table, rows)
        asked = idx >= -1
        if self.kind == "numeric":
            y = np.asarray(y, dtype=np.float64)
            asked &= ~np.isnan(y[:asked.shape[0]])
        yt = np.asarray(y)[:asked.shape[0]][asked]
        p = pred[asked]
        n = int(yt.shape[0])
        res = {"rows": n, "rows in the default rule": int(np.count_nonzero(idx[asked] == -1))}
        if n == 0:
            return res
        if self.kind == "binary":
            yt = yt.astype(bool)
            q = np.clip(p, 1e-6, 1 - 1e-6)
            bits_list = float(-np.sum(np.where(yt, np.log2(q), np.log2(1 - q))))
            q0 = min(max(self.default_statistic, 1e-6), 1 - 1e-6)
            bits_default = float(-np.sum(np.where(yt, math.log2(q0), math.log2(1 - q0))))
            res["log-loss list (bits/row)"] = bits_list / n
            res["log-loss default (bits/row)"] = bits_default / n
            res["bits saved on test"] = bits_default - bits_list
            res["accuracy at 0.5"] = float(np.mean((p >= 0.5) == yt))
            res["Brier score"] = float(np.mean((p - yt) ** 2))
            res["AUC"] = auc(p, yt)
        else:
            sd = np.maximum(spread[asked], self.default_spread * 1e-3)
            bits_list = float(np.sum(_gauss_bits_each(yt, p, sd)))
            bits_default = float(np.sum(_gauss_bits_each(yt, np.full(n, self.default_statistic),
                                                         np.full(n, self.default_spread))))
            res["bits saved on test"] = bits_default - bits_list
            res["RMSE list"] = float(math.sqrt(np.mean((yt - p) ** 2)))
            res["RMSE default"] = float(math.sqrt(np.mean((yt - self.default_statistic) ** 2)))
            res["MAE list"] = float(np.mean(np.abs(yt - p)))
        return res


def model_bits(conditions, nr_attributes, nr_bins=8):
    """L(S): bits to write the description down"""
    d = conditions.size()
    if d == 0:
        return 0.0
    bits = log_star(d)
    for i in range(d):
        c = conditions.get(i)
        col = c.get_column()
        bits += math.log2(max(nr_attributes, 1))
        t = col.get_type()
        op = c.get_operator()
        if t == AttributeType.NOMINAL:
            k = max(2, col.get_cardinality())
            if op == Operator.ELEMENT_OF:
                bits += k                    # one bit per value: in or out
            else:
                bits += math.log2(k)
        elif t == AttributeType.NUMERIC:
            k = max(2, min(col.get_cardinality(), max(nr_bins, 2)))
            if op == Operator.BETWEEN:
                bits += 2 * math.log2(k)
            else:
                bits += 1 + math.log2(k)     # which of <=, >= and which cut point
        else:
            bits += 1.0
    return bits


def _entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _gauss_bits(y, mu, sd):
    return float(np.sum(_gauss_bits_each(y, np.full(y.shape[0], mu), np.full(y.shape[0], sd))))


def _gauss_bits_each(y, mu, sd):
    """-log2 of the normal density (the precision constant cancels in every difference)"""
    return 0.5 * np.log2(2 * math.pi * sd * sd) + (y - mu) ** 2 / (2 * sd * sd * math.log(2.0))


def auc(scores, labels):
    """area under the ROC curve via the rank sum, ties count half"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    pos = int(np.count_nonzero(labels))
    neg = labels.shape[0] - pos
    if pos == 0 or neg == 0:
        return float('nan')
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    ranks = np.empty(s.shape[0], dtype=np.float64)
    i = 0
    while i < s.shape[0]:
        j = i
        while j + 1 < s.shape[0] and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((np.sum(r[labels]) - pos * (pos + 1) / 2.0) / (pos * neg))
