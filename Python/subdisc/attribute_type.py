from enum import Enum

from . import log

# used for file loading / Column.set_missing_value
_BOOLEAN_POSITIVES = {"1", "true"}
_BOOLEAN_NEGATIVES = {"0", "false"}
DEFAULT_BINARY_TRUE_STRING = "1"
DEFAULT_BINARY_FALSE_STRING = "0"


def _jlower_trim(s):
    from .java_compat import jtrim
    return jtrim(s.lower())


class AttributeType(Enum):
    """All available attribute types."""
    NOMINAL = 0
    NUMERIC = 1
    ORDINAL = 2
    BINARY = 3

    def __str__(self):
        return self.name.lower()

    @staticmethod
    def from_string(text):
        """Case insensitive lookup, NOMINAL when nothing fits."""
        for at in AttributeType:
            if str(at).lower() == (text or "").lower():
                return at
        log.log_command_line("'%s' is not a valid AttributeType. Returning '%s'." % (text, AttributeType.get_default()))
        return AttributeType.get_default()

    @staticmethod
    def is_valid_binary_true_value(value):
        return _jlower_trim(value) in _BOOLEAN_POSITIVES

    @staticmethod
    def is_valid_binary_false_value(value):
        return _jlower_trim(value) in _BOOLEAN_NEGATIVES

    @staticmethod
    def is_valid_binary_value(value):
        v = _jlower_trim(value)
        return v in _BOOLEAN_POSITIVES or v in _BOOLEAN_NEGATIVES

    @staticmethod
    def get_default():
        return AttributeType.NOMINAL
