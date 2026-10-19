from enum import Enum

from . import log


class TargetType(Enum):
    # when adding target types, check all the static methods below
    SINGLE_NOMINAL = (0, "single nominal")
    SINGLE_NUMERIC = (1, "single numeric")
    MULTI_NUMERIC = (2, "multi-numeric")
    SINGLE_ORDINAL = (3, "single ordinal")
    DOUBLE_REGRESSION = (4, "double regression")
    DOUBLE_CORRELATION = (5, "double correlation")
    DOUBLE_BINARY = (6, "double binary")
    SCAPE = (7, "soft classifier performance evaluation")
    MULTI_LABEL = (8, "multi-label")
    MULTI_BINARY_CLASSIFICATION = (9, "multi binary classification")
    LABEL_RANKING = (10, "label ranking")

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

    @staticmethod
    def from_string(text):
        for t in TargetType:
            if t.GUI_TEXT.lower() == (text or "").lower():
                return t
        log.log_command_line("'%s' is not a valid TargetType. Returning '%s'." % (text, TargetType.get_default().GUI_TEXT))
        return TargetType.get_default()

    @staticmethod
    def get_default():
        return TargetType.SINGLE_NOMINAL

    # one table instead of the switch statements, columns:
    # implemented, secondary target, multi targets, multi regression targets,
    # misc field, target attribute, target value, base model
    @staticmethod
    def _flags(t):
        return _FLAGS[t]

    @staticmethod
    def is_implemented(t):
        return _FLAGS[t][0]

    @staticmethod
    def has_secondary_target(t):
        """used by XMLNodeTargetConcept"""
        return _FLAGS[t][1]

    @staticmethod
    def has_multi_targets(t):
        return _FLAGS[t][2]

    @staticmethod
    def has_multi_regression_targets(t):
        return _FLAGS[t][3]

    @staticmethod
    def has_misc_field(t):
        return _FLAGS[t][4]

    @staticmethod
    def has_target_attribute(t):
        return _FLAGS[t][5]

    @staticmethod
    def has_target_value(t):
        return _FLAGS[t][6]

    @staticmethod
    def has_base_model(t):
        return _FLAGS[t][7]


#              impl   2nd    multi  mreg   misc   attr   value  model
_FLAGS = {
    TargetType.SINGLE_NOMINAL:              (True,  False, False, False, True,  True,  True,  False),
    TargetType.SINGLE_NUMERIC:              (True,  False, False, False, False, True,  False, True),
    TargetType.MULTI_NUMERIC:               (True,  False, True,  False, False, False, False, True),
    TargetType.SINGLE_ORDINAL:              (False, False, False, False, False, True,  False, False),
    TargetType.DOUBLE_REGRESSION:           (True,  True,  False, True,  True,  True,  False, True),
    TargetType.DOUBLE_CORRELATION:          (True,  True,  False, False, True,  True,  False, True),
    TargetType.DOUBLE_BINARY:               (True,  True,  False, False, True,  True,  False, False),
    TargetType.SCAPE:                       (True,  True,  False, False, True,  True,  False, True),
    TargetType.MULTI_LABEL:                 (True,  False, True,  False, False, False, False, True),
    TargetType.MULTI_BINARY_CLASSIFICATION: (False, False, True,  False, True,  False, False, False),
    TargetType.LABEL_RANKING:               (True,  False, False, False, False, True,  False, True),
}
