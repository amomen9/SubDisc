"""Immutable condition: Column, Operator and one value.

Java has five constructors, one per value type. Here there is one
constructor plus small factory functions so the call sites stay readable:
Condition.nominal(), .nominal_set(), .binary(), .interval(), .numeric().
"""

import math
from functools import total_ordering

from .attribute_type import AttributeType
from .java_compat import float_to_string, float_compare, parse_float, to_f32
from .operators import Operator
from .value_set import ValueSet
from .interval import Interval

# NOTE relies on the fact that MIN_VALUE is smaller than -MAX_VALUE,
# MAX_VALUE is the largest valid (sort) index that can occur
UNINITIALISED_SORT_INDEX = -2147483648

_NOMINAL, _NOMINAL_SET, _BINARY, _INTERVAL, _NUMERIC = range(5)


def _exception(pre, post):
    return ValueError(pre + " can not be " + post)


@total_ordering
class Condition:
    __slots__ = ("_column", "_operator", "_sort_index", "_nominal_value", "_nominal_value_set",
                 "_numeric_value", "_interval", "_binary_value", "_kind")

    def __init__(self, condition_base, value, sort_index=None):
        """value decides which kind of condition this is, like the java overloads:
        str -> nominal EQUALS, ValueSet -> nominal ELEMENT_OF, bool -> binary,
        Interval -> numeric BETWEEN, float (+ sort_index) -> numeric =, <=, >=
        """
        column = condition_base.get_column()
        op = condition_base.get_operator()
        ctype = column.get_type()

        self._column = column
        self._operator = op
        self._sort_index = UNINITIALISED_SORT_INDEX
        self._nominal_value = None
        self._nominal_value_set = None
        self._numeric_value = float('nan')
        self._interval = None
        self._binary_value = False

        if isinstance(value, bool):
            if ctype != AttributeType.BINARY:
                raise _exception("ConditionBase.Column.AttributeType", str(ctype))
            assert op == Operator.EQUALS
            self._binary_value = value
            self._kind = _BINARY
        elif isinstance(value, str):
            if ctype != AttributeType.NOMINAL:
                raise _exception("ConditionBase.Column.AttributeType", str(ctype))
            if op != Operator.EQUALS:
                raise _exception("ConditionBase.Operator", op.GUI_TEXT)
            self._nominal_value = value
            self._kind = _NOMINAL
        elif isinstance(value, ValueSet):
            if ctype != AttributeType.NOMINAL:
                raise _exception("ConditionBase.Column.AttributeType", str(ctype))
            if op != Operator.ELEMENT_OF:
                raise _exception("ConditionBase.Operator", op.GUI_TEXT)
            self._nominal_value_set = value
            self._kind = _NOMINAL_SET
        elif isinstance(value, Interval):
            if ctype != AttributeType.NUMERIC:
                raise _exception("ConditionBase.Column.AttributeType", str(ctype))
            if op != Operator.BETWEEN:
                raise _exception("ConditionBase.Operator", op.GUI_TEXT)
            self._interval = value
            self._kind = _INTERVAL
        elif value is None:
            raise _exception("value", "null")
        else:
            if ctype != AttributeType.NUMERIC:
                raise _exception("ConditionBase.Column.AttributeType", str(ctype))
            if op == Operator.BETWEEN:
                raise _exception("ConditionBase.Operator", op.GUI_TEXT)
            if op not in (Operator.EQUALS, Operator.LESS_THAN_OR_EQUAL, Operator.GREATER_THAN_OR_EQUAL):
                raise AssertionError(op)
            v = to_f32(value)
            # NaN is not allowed as a condition value (it is as a data value)
            if math.isnan(v):
                raise _exception("float value", "NaN")
            self._numeric_value = v
            self._sort_index = UNINITIALISED_SORT_INDEX if sort_index is None else int(sort_index)
            self._kind = _NUMERIC

    def get_column(self):
        return self._column

    def get_operator(self):
        return self._operator

    def get_sort_index(self):
        return self._sort_index

    # no type validity checks are performed
    def get_nominal_value(self):
        return self._nominal_value

    def get_nominal_value_set(self):
        return self._nominal_value_set

    def get_numeric_value(self):
        return self._numeric_value

    def get_numeric_interval(self):
        return self._interval

    def get_binary_value(self):
        return self._binary_value

    def get_value(self):
        """Value as shown in the gui and result files."""
        ctype = self._column.get_type()
        if ctype == AttributeType.NOMINAL:
            if self._operator == Operator.EQUALS:
                return "'" + self._nominal_value + "'"
            if self._operator == Operator.ELEMENT_OF:
                return str(self._nominal_value_set)
            raise AssertionError(AttributeType.NOMINAL)
        if ctype == AttributeType.NUMERIC:
            if self._operator in (Operator.EQUALS, Operator.LESS_THAN_OR_EQUAL, Operator.GREATER_THAN_OR_EQUAL):
                return float_to_string(self._numeric_value)
            if self._operator == Operator.BETWEEN:
                return str(self._interval)
            raise AssertionError(AttributeType.NUMERIC)
        if ctype == AttributeType.BINARY:
            return "'1'" if self._binary_value else "'0'"
        raise AssertionError(ctype)

    def __str__(self):
        return self._column.get_name() + " " + str(self._operator) + " " + self.get_value()

    __repr__ = __str__

    def compare_to(self, other):
        if self is other:
            return 0
        # Conditions about a Column with smaller index come first
        cmp = self._column.get_index() - other._column.get_index()
        if cmp != 0:
            return cmp
        # same column, check operator
        cmp = self._operator.ordinal - other._operator.ordinal
        if cmp != 0:
            return cmp
        ctype = self._column.get_type()
        if ctype == AttributeType.NOMINAL:
            if self._nominal_value is not None:
                a, b = self._nominal_value, other._nominal_value
                return -1 if a < b else (1 if a > b else 0)
            return self._nominal_value_set.compare_to(other._nominal_value_set)
        if ctype == AttributeType.NUMERIC:
            if self._interval is not None:
                return self._interval.compare_to(other._interval)
            # NOTE considers 0.0 to be greater than -0.0
            return float_compare(self._numeric_value, other._numeric_value)
        if ctype == AttributeType.BINARY:
            if not self._binary_value:
                return -1 if other._binary_value else 0
            return 0 if other._binary_value else 1
        raise AssertionError("ERROR: compareTo()\n%s\n%s" % (self, other))

    def __lt__(self, other):
        return self.compare_to(other) < 0

    def __eq__(self, other):
        # java Conditions do not override equals(), identity it is
        return self is other

    def __hash__(self):
        return id(self)

    def strictly_specialises(self, other):
        """True when all is the same, numeric, and this value is tighter
        (a <= 10 specialises a <= 20)."""
        if self.get_column() is not other.get_column():
            return False
        if self.get_operator() != other.get_operator():
            return False
        # the java code compares the two value Strings with == here, which
        # never holds for two freshly built Strings, so no check
        if self.get_column().get_type() != AttributeType.NUMERIC or other.get_column().get_type() != AttributeType.NUMERIC:
            return False
        a = parse_float(self.get_value())
        b = parse_float(other.get_value())
        if self.get_operator() == Operator.LESS_THAN_OR_EQUAL and a >= b:
            return False
        if self.get_operator() == Operator.GREATER_THAN_OR_EQUAL and a <= b:
            return False
        return True

    def logically_equivalent(self, other):
        if self.get_column() is not other.get_column():
            return False
        if self.get_operator() != other.get_operator():
            return False
        return self.get_value() == other.get_value()
