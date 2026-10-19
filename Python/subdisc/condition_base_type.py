from enum import Enum


class ConditionBaseType(Enum):
    """All valid ConditionBase combinations of Column type and Operator."""
    NOMINAL_ELEMENT_OF = 0
    NOMINAL_EQUALS = 1
    NUMERIC_EQUALS = 2
    NUMERIC_LESS_THAN_OR_EQUAL = 3
    NUMERIC_GREATER_THAN_OR_EQUAL = 4
    NUMERIC_BETWEEN = 5
    BINARY_EQUALS = 6
