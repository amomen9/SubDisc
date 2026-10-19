class ConditionBase:
    """A Column together with an Operator, no value yet."""

    __slots__ = ("_column", "_operator")

    def __init__(self, column, operator):
        if column is None:
            raise ValueError("ConditionBase: Column can not be null")
        if operator is None:
            raise ValueError("ConditionBase: Operator can not be null")
        if not operator.is_valid_for(column.get_type()):
            raise ValueError("ConditionBase: Operator '%s' not valid for Column of AttributeType '%s'\n"
                             % (operator.GUI_TEXT, column.get_type()))
        self._column = column
        self._operator = operator

    def get_column(self):
        return self._column

    def get_operator(self):
        return self._operator

    def __str__(self):
        return self._column.get_name() + " " + self._operator.GUI_TEXT
