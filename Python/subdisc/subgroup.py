"""A Subgroup: a ConditionList, the rows it covers and its quality.

Not thread safe, like the java class (the gui runs one search at a time).
"""

import threading
from functools import total_ordering

import numpy as np

from . import condition_list_builder as clb
from .java_compat import float_compare, div


@total_ordering
class Subgroup:
    __slots__ = ("_conditions", "_coverage", "_parent_set", "_has_quality", "_measure_value",
                 "_secondary_statistic", "_tertiary_statistic", "_members", "_members_lock", "_id",
                 "_is_p_value_computed", "_p_value", "_dag", "_label_ranking", "_label_ranking_matrix",
                 "_regression_model", "__weakref__")

    def __init__(self, conditions=None, members=None, subgroup_set=None, *, _raw=False):
        """Subgroup(conditions, members, subgroup_set); members needs at least one row.
        Subgroup(None, members) is the Validation-only form."""
        self._members_lock = threading.RLock()
        self._has_quality = False
        self._measure_value = 0.0
        self._secondary_statistic = 0.0
        self._tertiary_statistic = 0.0
        self._id = 0
        self._is_p_value_computed = False
        self._p_value = float('nan')
        self._dag = None
        self._label_ranking = None
        self._label_ranking_matrix = None
        self._regression_model = None
        self._conditions = conditions
        self._parent_set = subgroup_set
        self._members = None
        self._coverage = 0
        if _raw:
            return
        if members is None:
            raise ValueError("arguments can not be null")
        if conditions is not None and subgroup_set is None:
            raise ValueError("arguments can not be null")
        if not members.any():
            raise ValueError("Subgroups must have members")
        self._init_members(members)

    def _init_members(self, members):
        with self._members_lock:
            # TODO MM (java) a copy would be safer
            self._members = members
            self._coverage = int(np.count_nonzero(members))

    @staticmethod
    def _child(parent, condition):
        s = Subgroup(_raw=True)
        s._conditions = clb.create_list(parent._conditions, condition)
        s._parent_set = parent._parent_set
        return s

    def get_refined_subgroup(self, condition, new_members=None, coverage=None):
        """New Subgroup with condition added; members are evaluated unless given."""
        if condition is None:
            raise ValueError("arguments can not be null")
        s = Subgroup._child(self, condition)
        if new_members is not None:
            s._coverage = coverage if coverage is not None else int(np.count_nonzero(new_members))
            s._members = new_members
            return s
        # see NOTE in java: the parent scores are copied, never used
        s._measure_value = self._measure_value
        s._secondary_statistic = self._secondary_statistic
        s._tertiary_statistic = self._tertiary_statistic
        parent_members = self._get_members_unsafe()
        s._init_members(condition.get_column().evaluate(parent_members, condition))
        return s

    def get_refined_subgroup_scored(self, condition, quality, secondary, tertiary, child_coverage):
        """for direct computation of quality scores, no members evaluated"""
        s = Subgroup._child(self, condition)
        s._coverage = child_coverage
        s._has_quality = True
        s._measure_value = quality
        s._secondary_statistic = secondary
        s._tertiary_statistic = tertiary
        return s

    def _get_members_unsafe(self):
        with self._members_lock:
            if self._members is None:
                b = self._parent_set.get_all_data_bit_set_clone()
                for c in self._conditions:
                    b = c.get_column().evaluate(b, c)
                self._members = b
            return self._members

    def get_conditions(self):
        return self._conditions

    def get_depth(self):
        return self._conditions.size()

    def get_coverage(self):
        return self._coverage

    def get_parent_set(self):
        return self._parent_set

    def has_quality(self):
        return self._has_quality

    def get_measure_value(self):
        return self._measure_value

    def set_measure_value(self, v):
        self._measure_value = float(v)
        self._has_quality = True

    def get_secondary_statistic(self):
        return self._secondary_statistic

    def set_secondary_statistic(self, v):
        self._secondary_statistic = float(v)

    def get_tertiary_statistic(self):
        return self._tertiary_statistic

    def set_tertiary_statistic(self, v):
        self._tertiary_statistic = float(v)

    def get_members(self):
        """a copy, changing it does not change the Subgroup"""
        return self._get_members_unsafe().copy()

    def members_view(self):
        """no copy, for read only use in the mining code"""
        return self._get_members_unsafe()

    def kill_members(self):
        """CandidateQueue frees the memory, members are re-evaluated when needed"""
        with self._members_lock:
            self._members = None

    def revive_members(self):
        self._get_members_unsafe()

    def covers(self, row):
        return bool(self._get_members_unsafe()[row])

    def get_id(self):
        return self._id

    def set_id(self, i):
        self._id = int(i)

    def get_p_value(self):
        return self._p_value if self._is_p_value_computed else float('nan')

    def set_p_value(self, value):
        """a float (fake p-value) or a NormalDistribution"""
        self._is_p_value_computed = True
        if hasattr(value, "calc_cdf"):
            self._p_value = 1.0 - value.calc_cdf(self._measure_value)
        else:
            self._p_value = float(value)

    def set_empirical_p_value(self, qualities):
        self._is_p_value_computed = True
        n = len(qualities)
        p = 0.0
        for q in qualities:
            if q >= self._measure_value:
                p += 1
        self._p_value = div(p, n)

    def renounce_p_value(self):
        self._is_p_value_computed = False

    # model specific information
    def set_dag(self, dag):
        self._dag = dag

    def get_dag(self):
        return self._dag

    def set_label_ranking(self, lr):
        self._label_ranking = lr

    def get_label_ranking(self):
        return self._label_ranking

    def set_label_ranking_matrix(self, m):
        self._label_ranking_matrix = m

    def get_label_ranking_matrix(self):
        return self._label_ranking_matrix

    def get_regression_model(self):
        return self._regression_model

    def set_regression_model(self, m):
        self._regression_model = m

    def count_common(self, other_members):
        return int(np.count_nonzero(self._get_members_unsafe() & other_members))

    def get_true_positive_rate(self):
        target = self._parent_set.get_binary_target_clone()
        if target is None:
            return 0.0
        head_body = int(np.count_nonzero(target & self._get_members_unsafe()))
        total_target = self._parent_set.get_total_target_coverage()
        if total_target <= 0:
            raise AssertionError()
        return head_body / total_target

    def get_false_positive_rate(self):
        target = self._parent_set.get_binary_target_clone()
        if target is None:
            return 0.0
        head_body = int(np.count_nonzero(target & self._get_members_unsafe()))
        total = self._parent_set.get_total_coverage()
        total_target = self._parent_set.get_total_target_coverage()
        not_head = total - total_target
        if total <= 0 or total_target < 0 or total < total_target or not_head < 0:
            raise AssertionError()
        # (FP / !H)
        return div(self._coverage - head_body, not_head)

    def compare_to(self, other):
        """better score first, then larger coverage, then the ConditionList"""
        if self is other:
            return 0
        cmp = float_compare(self._measure_value, other._measure_value)
        if cmp != 0:
            return -cmp
        cmp = self._coverage - other._coverage
        if cmp != 0:
            return -1 if cmp > 0 else 1
        return self._conditions.compare_to(other._conditions)

    def __lt__(self, other):
        return self.compare_to(other) < 0

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def __str__(self):
        return str(self._conditions)

    __repr__ = __str__

    def strictly_specialises(self, other):
        return self._conditions.strictly_specialises(other._conditions)
