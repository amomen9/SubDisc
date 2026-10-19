from enum import Enum

from .attribute_type import AttributeType


class Operator(Enum):
    # value = (ordinal, gui text); ELEMENT_OF and BETWEEN share "in" so the
    # ordinal keeps them apart
    ELEMENT_OF = (0, "in")
    # XXX DEPTH_FIRST relies on EQUALS being before LEQ / GEQ
    EQUALS = (1, "=")
    LESS_THAN_OR_EQUAL = (2, "<=")
    GREATER_THAN_OR_EQUAL = (3, ">=")
    BETWEEN = (4, "in")

    @property
    def GUI_TEXT(self):
        return self.value[1]

    @property
    def ordinal(self):
        return self.value[0]

    def __str__(self):
        return self.value[1]

    def __lt__(self, other):
        return self.value[0] < other.value[0]

    def is_valid_for(self, attribute_type):
        if self is Operator.ELEMENT_OF:
            return attribute_type == AttributeType.NOMINAL
        if self is Operator.EQUALS:
            # ORDINAL is not implemented, do not use
            return attribute_type in (AttributeType.NOMINAL, AttributeType.NUMERIC, AttributeType.BINARY)
        return attribute_type == AttributeType.NUMERIC

    @staticmethod
    def from_string(text):
        for o in Operator:
            if o.GUI_TEXT == text:
                return o
        raise ValueError("Operator.formString(), unknown Operator: " + str(text))

    @staticmethod
    def set():
        return frozenset(Operator)

    @staticmethod
    def get_operators(attribute_type):
        return [o for o in Operator if o.is_valid_for(attribute_type)]

    def is_simple(self):
        """Is it =, <= or >= ?"""
        return self in (Operator.EQUALS, Operator.LESS_THAN_OR_EQUAL, Operator.GREATER_THAN_OR_EQUAL)
