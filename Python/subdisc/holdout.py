"""Holdout testing of a mining result (not in the java version).

A subgroup is a description (age >= 33 AND sex = 'Male'), so it can be
applied to rows it was not mined on. This module does that in two ways:

- on the rows of the same table that were held out of the mining run
  (random_split() makes such a split, the GUI then mines on the train rows)
- on a second file with the same columns (load_test_table())

and scores every subgroup there with the same quality measure, so the
train and test numbers can be put next to each other.
"""

import copy
import math

import numpy as np

from . import log
from .attribute_type import AttributeType
from .column import Column
from .operators import Operator
from .subgroup import Subgroup
from .table import Table
from .target_type import TargetType


# ----------------------------------------------------------------------
# splitting
# ----------------------------------------------------------------------

def random_split(nr_rows, test_fraction, seed=1, within=None):
    """(train, test) row masks; within limits the split to those rows"""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test fraction must be between 0 and 1, not %s" % test_fraction)
    pool = np.arange(nr_rows) if within is None else np.flatnonzero(within[:nr_rows])
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    nr_test = int(round(test_fraction * pool.shape[0]))
    test = np.zeros(nr_rows, dtype=bool)
    test[pool[:nr_test]] = True
    train = np.zeros(nr_rows, dtype=bool)
    train[pool[nr_test:]] = True
    return train, test


# ----------------------------------------------------------------------
# a second file with the same columns
# ----------------------------------------------------------------------

def load_test_table(train_table, path):
    """The data of path in a Table with the columns (names, order, types) of
    train_table. Returns (table, warnings); raises ValueError when the
    columns do not match."""
    from .file_handler import FileHandler
    t = Table(path)
    for c in train_table.get_columns():
        nc = Column(c.get_name(), c.get_short(), c.get_type(), c.get_index(), 0)
        nc.set_is_enabled(c.get_is_enabled())
        t.get_columns().append(nc)
    t._nr_columns = len(t.get_columns())
    FileHandler.populate_from_file(path, t, False)
    t.update()
    if t.get_nr_rows() == 0:
        raise ValueError("No rows were read from '%s'. The test file needs the same columns, in the same order, "
                         "as the training data (%s)." % (path, ", ".join(c.get_name() for c in train_table.get_columns())))
    warnings = []
    for a, b in zip(train_table.get_columns(), t.get_columns()):
        if a.get_type() != b.get_type():
            warnings.append("column '%s' is %s in the training data but %s in the test file" % (a.get_name(), a.get_type(), b.get_type()))
    return t, warnings


# ----------------------------------------------------------------------
# applying a description to any table
# ----------------------------------------------------------------------

def _condition_mask(column, condition):
    """rows of column for which condition holds; condition may belong to a
    column of another table with the same name, the logic follows
    Column.evaluate() (missing values never satisfy a condition)"""
    n = column.size()
    op = condition.get_operator()
    missing = np.asarray(column.missing_view()[:n], dtype=bool)
    ct = column.get_type()
    kind = condition.get_column().get_type()
    if kind == AttributeType.NOMINAL:
        values = _as_strings(column)
        if op == Operator.EQUALS:
            return (values == condition.get_nominal_value()) & ~missing
        if op == Operator.ELEMENT_OF:
            vs = condition.get_nominal_value_set()
            return np.array([vs.contains(v) for v in values], dtype=bool) & ~missing
        raise AssertionError(op)
    if kind == AttributeType.NUMERIC:
        f = _as_floats(column)
        if op == Operator.BETWEEN:
            return condition.get_numeric_interval().between_mask(f) & ~missing & ~np.isnan(f)
        v = np.float32(condition.get_numeric_value())
        if op == Operator.EQUALS:
            res = f == v
        elif op == Operator.LESS_THAN_OR_EQUAL:
            res = f <= v
        elif op == Operator.GREATER_THAN_OR_EQUAL:
            res = f >= v
        else:
            raise AssertionError(op)
        return res & ~missing
    if kind == AttributeType.BINARY:
        if ct == AttributeType.BINARY:
            b = np.asarray(column.get_binaries()[:n], dtype=bool)
        else:
            s = _as_strings(column)
            b = np.array([AttributeType.is_valid_binary_value(x) and AttributeType.is_valid_binary_true_value(x)
                          for x in s], dtype=bool)
        return b if condition.get_binary_value() else ~b
    raise AssertionError(kind)


def _as_strings(column):
    n = column.size()
    if column.get_type() == AttributeType.NOMINAL:
        dv = np.array(list(column.distinct_values()) or [""], dtype=object)
        return dv[np.asarray(column.nominal_codes_view()[:n])]
    return np.array([column.get_string(i) for i in range(n)], dtype=object)


def _as_floats(column):
    n = column.size()
    if column.get_type() == AttributeType.NUMERIC:
        return np.asarray(column.floats_view()[:n], dtype=np.float32)
    if column.get_type() == AttributeType.BINARY:
        return np.asarray(column.get_binaries()[:n], dtype=np.float32)
    out = np.full(n, np.nan, dtype=np.float32)
    for i, s in enumerate(_as_strings(column)):
        try:
            out[i] = float(s)
        except (TypeError, ValueError):
            pass
    return out


def evaluate_conditions(table, conditions, base=None):
    """rows of table covered by the ConditionList (columns found by name),
    limited to base when given"""
    n = table.get_nr_rows()
    members = np.ones(n, dtype=bool) if base is None else np.asarray(base[:n], dtype=bool).copy()
    for i in range(conditions.size()):
        c = conditions.get(i)
        col = table.get_column(c.get_column().get_name())
        if col is None:
            raise ValueError("column '%s' is not in table '%s'" % (c.get_column().get_name(), table.get_name()))
        m = _condition_mask(col, c)
        members[:m.shape[0]] &= m
        if m.shape[0] < n:
            members[m.shape[0]:] = False
    return members


# ----------------------------------------------------------------------
# scoring subgroups on other rows
# ----------------------------------------------------------------------

def map_search_parameters(search_parameters, table):
    """a copy of the search parameters whose target concept points to the
    columns of table with the same names"""
    sp = copy.copy(search_parameters)
    tc = copy.copy(search_parameters.get_target_concept())

    def same(c):
        if c is None:
            return None
        other = table.get_column(c.get_name())
        if other is None:
            raise ValueError("target column '%s' is not in table '%s'" % (c.get_name(), table.get_name()))
        return other

    tc.set_primary_target(same(tc.get_primary_target()))
    tc.set_secondary_target(same(tc.get_secondary_target()))
    if tc.get_multi_targets():
        tc.set_multi_targets([same(c) for c in tc.get_multi_targets()])
    sp.set_target_concept(tc)
    return sp


class HoldoutRow:
    """one subgroup, train and test side by side"""
    __slots__ = ("subgroup", "train_coverage", "train_quality", "train_secondary",
                 "test_coverage", "test_quality", "test_secondary", "test_tertiary", "test_members")

    def drop(self):
        return self.train_quality - self.test_quality


def score_on(subgroups, search_parameters, eval_table, eval_rows=None):
    """HoldoutRow per subgroup: its description applied to eval_rows of
    eval_table (None: all rows) and scored with the quality measure of
    search_parameters, the 'base' statistics come from eval_rows only."""
    from .process import create_subgroup_discovery
    sp = map_search_parameters(search_parameters, eval_table)
    n = eval_table.get_nr_rows()
    base = np.ones(n, dtype=bool) if eval_rows is None else np.asarray(eval_rows[:n], dtype=bool)
    mem = log.COMMANDLINELOG
    log.COMMANDLINELOG = False
    try:
        sd = create_subgroup_discovery(eval_table, None if eval_rows is None else base.copy(), sp, None)
    finally:
        log.COMMANDLINELOG = mem
    out = []
    for s in subgroups:
        r = HoldoutRow()
        r.subgroup = s
        r.train_coverage = s.get_coverage()
        r.train_quality = float(s.get_measure_value())
        r.train_secondary = float(s.get_secondary_statistic())
        members = evaluate_conditions(eval_table, s.get_conditions(), base)
        r.test_members = members
        r.test_coverage = int(np.count_nonzero(members))
        r.test_quality = float('nan')
        r.test_secondary = float('nan')
        r.test_tertiary = float('nan')
        if r.test_coverage > 0:
            child = Subgroup(s.get_conditions(), members, sd.get_result())
            mem = log.COMMANDLINELOG
            log.COMMANDLINELOG = False
            try:
                r.test_quality = float(sd._evaluate_candidate(child))
            finally:
                log.COMMANDLINELOG = mem
            r.test_secondary = float(child.get_secondary_statistic())
            r.test_tertiary = float(child.get_tertiary_statistic())
        out.append(r)
    return out


def summary(rows, nr_train_rows, nr_test_rows, quality_minimum=None):
    """a few numbers about how well the result carries over"""
    ok = [r for r in rows if not math.isnan(r.test_quality)]
    res = {
        "subgroups": len(rows),
        "empty on test": len(rows) - len(ok),
        "train rows": nr_train_rows,
        "test rows": nr_test_rows,
    }
    if ok:
        tr = np.array([r.train_quality for r in ok])
        te = np.array([r.test_quality for r in ok])
        res["mean train quality"] = float(tr.mean())
        res["mean test quality"] = float(te.mean())
        res["mean drop"] = float((tr - te).mean())
        res["rank correlation"] = spearman(tr, te)
        if quality_minimum is not None:
            res["still above the quality minimum"] = int(np.count_nonzero(te > quality_minimum))
        cov_train = np.array([r.train_coverage / max(nr_train_rows, 1) for r in ok])
        cov_test = np.array([r.test_coverage / max(nr_test_rows, 1) for r in ok])
        res["mean coverage share train"] = float(cov_train.mean())
        res["mean coverage share test"] = float(cov_test.mean())
    return res


def _ranks(x):
    """average ranks, ties share the mean rank"""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape[0] < 2:
        return float('nan')
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float(np.sum(ra * ra) * np.sum(rb * rb)))
    return float(np.sum(ra * rb) / den) if den > 0 else float('nan')


def target_values(table, target_concept):
    """the target per row of table: bool (single nominal: is it the target
    value) or float with NaN for missing (single numeric)"""
    from .condition import Condition
    from .condition_base import ConditionBase
    n = table.get_nr_rows()
    tt = target_concept.get_target_type()
    col = target_concept.get_primary_target()
    if tt == TargetType.SINGLE_NOMINAL:
        value = target_concept.get_target_value()
        cb = ConditionBase(col, Operator.EQUALS)
        if col.get_type() == AttributeType.BINARY:
            cond = Condition(cb, AttributeType.is_valid_binary_true_value(value))
        else:
            if value not in list(col.distinct_values()):
                return np.zeros(n, dtype=bool)
            cond = Condition(cb, value)
        return np.asarray(col.evaluate(np.ones(n, dtype=bool), cond)[:n], dtype=bool)
    if tt == TargetType.SINGLE_NUMERIC:
        f = np.asarray(col.floats_view()[:n], dtype=np.float64).copy()
        f[np.asarray(col.missing_view()[:n], dtype=bool)] = np.nan
        return f
    raise ValueError("no single target for %s" % tt)


def target_kind(search_parameters):
    """'binary', 'numeric' or None: what a subgroup list can predict"""
    tt = search_parameters.get_target_type()
    if tt == TargetType.SINGLE_NOMINAL:
        return "binary"
    if tt == TargetType.SINGLE_NUMERIC:
        return "numeric"
    return None
