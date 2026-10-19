from enum import Enum

from . import log
from .operators import Operator


class NumericOperatorSetting(Enum):
    # (ordinal, gui text, plain text for Qt widgets)
    NORMAL = (0, "<html>&#8804;, &#8805;</html>", "≤, ≥")
    LEQ = (1, "<html>&#8804;</html>", "≤")
    GEQ = (2, "<html>&#8805;</html>", "≥")
    ALL = (3, "<html>&#8804;, &#8805;, =</html>", "≤, ≥, =")
    EQ = (4, "=", "=")
    INTERVALS = (5, "in", "in")

    @property
    def GUI_TEXT(self):
        return self.value[1]

    @property
    def DISPLAY_TEXT(self):
        return self.value[2]

    def __str__(self):
        return self.value[1]

    def get_operators(self):
        # XXX leave EQUALS first, see SubgroupDiscovery.createsRedundantEquals()
        return list(_OPERATORS[self])

    def includes_equals(self):
        return self in (NumericOperatorSetting.ALL, NumericOperatorSetting.EQ)

    @staticmethod
    def from_string(text):
        for n in NumericOperatorSetting:
            if n.GUI_TEXT.lower() == (text or "").lower() or n.DISPLAY_TEXT == text:
                return n
        log.log_command_line("'%s' is not a valid NumericOperators. Returning '%s'." % (text, NumericOperatorSetting.get_default().GUI_TEXT))
        return NumericOperatorSetting.get_default()

    @staticmethod
    def get_normal_values():
        # list, the order is the order in the gui; no intervals
        return [NumericOperatorSetting.NORMAL, NumericOperatorSetting.LEQ, NumericOperatorSetting.GEQ,
                NumericOperatorSetting.ALL, NumericOperatorSetting.EQ]

    @staticmethod
    def get_default():
        return NumericOperatorSetting.NORMAL

    @staticmethod
    def check(no, operator):
        """deprecated, only used by the old RefinementList constructor"""
        return operator in _OPERATORS[no]


_OPERATORS = {
    NumericOperatorSetting.NORMAL: (Operator.LESS_THAN_OR_EQUAL, Operator.GREATER_THAN_OR_EQUAL),
    NumericOperatorSetting.LEQ: (Operator.LESS_THAN_OR_EQUAL,),
    NumericOperatorSetting.GEQ: (Operator.GREATER_THAN_OR_EQUAL,),
    NumericOperatorSetting.ALL: (Operator.EQUALS, Operator.LESS_THAN_OR_EQUAL, Operator.GREATER_THAN_OR_EQUAL),
    NumericOperatorSetting.EQ: (Operator.EQUALS,),
    NumericOperatorSetting.INTERVALS: (Operator.BETWEEN,),
}


def operators_to_string(ops):
    """EnumSet.toString() of the operators, e.g. [<=, >=]"""
    return "[" + ", ".join(str(o) for o in sorted(ops)) + "]"
