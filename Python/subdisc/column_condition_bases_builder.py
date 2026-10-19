"""Per column, the ConditionBases the search refines with.

Nominal classes depend on the Operator, numeric ones on the
NumericOperatorSetting. Columns that are disabled, targets, or have a
cardinality <= 1 give no useful refinements and are left out.
"""

from .attribute_type import AttributeType
from .condition_base import ConditionBase
from .numeric_operator_setting import NumericOperatorSetting
from .operators import Operator
from .target_type import TargetType


class ColumnConditionBases:
    def get(self, index):
        raise NotImplementedError

    def first(self):
        """first non-None ConditionBase (any of the three for numeric)"""
        for i in range(3):
            cb = self.get(i)
            if cb is not None:
                return cb
        return None


class ColumnConditionBasesBinary(ColumnConditionBases):
    """assumes EQUALS is the only BINARY Operator"""
    __slots__ = ("_cb",)

    def __init__(self, column):
        self._cb = ConditionBase(column, Operator.EQUALS)

    def get(self, index):
        return self._cb


class ColumnConditionBasesNominalElementOf(ColumnConditionBases):
    __slots__ = ("_cb",)

    def __init__(self, column):
        self._cb = ConditionBase(column, Operator.ELEMENT_OF)

    def get(self, index):
        return self._cb


class ColumnConditionBasesNominalEquals(ColumnConditionBases):
    __slots__ = ("_cb",)

    def __init__(self, column):
        self._cb = ConditionBase(column, Operator.EQUALS)

    def get(self, index):
        return self._cb


class ColumnConditionBasesNumericRegular(ColumnConditionBases):
    """index 0: EQUALS (or BETWEEN for (best-)bins), 1: <=, 2: >="""
    __slots__ = ("_eq", "_leq", "_geq")

    def __init__(self, column, equals, use_leq, use_geq):
        self._eq = ConditionBase(column, equals) if equals is not None else None
        self._leq = ConditionBase(column, Operator.LESS_THAN_OR_EQUAL) if use_leq else None
        self._geq = ConditionBase(column, Operator.GREATER_THAN_OR_EQUAL) if use_geq else None

    def get(self, index):
        assert index in (0, 1, 2)
        return self._eq if index == 0 else (self._leq if index == 1 else self._geq)


class ColumnConditionBasesNumericIntervals(ColumnConditionBases):
    __slots__ = ("_cb",)

    def __init__(self, column):
        self._cb = ConditionBase(column, Operator.BETWEEN)

    def get(self, index):
        return self._cb


def get_column_condition_bases_set(table, search_parameters):
    tc = search_parameters.get_target_concept()
    use_sets = search_parameters.get_nominal_sets()
    # set-valued only for SINGLE_NOMINAL
    assert not use_sets or tc.get_target_type() == TargetType.SINGLE_NOMINAL
    n = search_parameters.get_numeric_operator_setting()
    is_interval = n == NumericOperatorSetting.INTERVALS
    ops = n.get_operators()
    use_equals = (not is_interval) and Operator.EQUALS in ops
    use_leq = (not is_interval) and Operator.LESS_THAN_OR_EQUAL in ops
    use_geq = (not is_interval) and Operator.GREATER_THAN_OR_EQUAL in ops
    # special case NUMERIC_EQ with (BEST_)BINS: BETWEEN
    s = search_parameters.get_numeric_strategy()
    e = None if not use_equals else (Operator.BETWEEN if s.is_discretiser() else Operator.EQUALS)

    result = []
    for c in table.get_columns():
        if not c.get_is_enabled() or tc.is_target_attribute(c):
            continue
        if c.get_cardinality() <= 1:
            continue
        t = c.get_type()
        if t == AttributeType.NOMINAL:
            result.append(ColumnConditionBasesNominalElementOf(c) if use_sets else ColumnConditionBasesNominalEquals(c))
        elif t == AttributeType.NUMERIC:
            if not is_interval:
                result.append(ColumnConditionBasesNumericRegular(c, e, use_leq, use_geq))
            else:
                result.append(ColumnConditionBasesNumericIntervals(c))
        elif t == AttributeType.BINARY:
            result.append(ColumnConditionBasesBinary(c))
        else:
            raise AssertionError(t)
    return tuple(result)
