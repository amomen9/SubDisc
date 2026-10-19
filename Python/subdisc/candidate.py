from .java_compat import float_compare
from .search_strategy import SearchStrategy


class Candidate:
    """A Subgroup waiting in the CandidateQueue, ordered by priority."""

    __slots__ = ("_subgroup", "_priority")

    # FIXME MM (java) disallow Candidates of size 0 and 1, they can not be refined
    def __init__(self, subgroup):
        self._subgroup = subgroup
        self._priority = subgroup.get_measure_value()

    def compare_to(self, other):
        if self is other:
            return 0
        # higher priority first
        cmp = float_compare(self._priority, other._priority)
        if cmp != 0:
            return -cmp
        # equal priorities; since r2174 two equivalent ConditionLists (same
        # Conditions, other search order) give 0, which removes redundancy
        return self._subgroup.compare_to(other._subgroup)

    def __lt__(self, other):
        return self.compare_to(other) < 0

    def get_priority(self):
        return self._priority

    def set_priority(self, p):
        self._priority = float(p)

    def get_subgroup(self):
        return self._subgroup

    def __str__(self):
        from .java_compat import double_to_string
        return "Candidate: priority=" + double_to_string(self._priority) + " ConditionsList=" + str(self._subgroup)

    @staticmethod
    def get_comparator(search_strategy):
        if search_strategy == SearchStrategy.DEPTH_FIRST:
            return compare_depth_first
        if search_strategy == SearchStrategy.BREADTH_FIRST:
            return compare_breadth_first
        raise AssertionError(search_strategy)


def compare_best_first(x, y):
    """the natural ordering, provided for completeness"""
    return 0 if x is y else x.compare_to(y)


def compare_depth_first(x, y):
    if x is y:
        return 0
    # deeper first, then ConditionList (depth 0 needs that ordering)
    cmp = x._subgroup.get_depth() - y._subgroup.get_depth()
    if cmp != 0:
        return -cmp
    return x._subgroup.get_conditions().compare_to(y._subgroup.get_conditions())


def compare_breadth_first(x, y):
    if x is y:
        return 0
    # lower depth first, then normal comparison
    cmp = x._subgroup.get_depth() - y._subgroup.get_depth()
    return cmp if cmp != 0 else x.compare_to(y)
