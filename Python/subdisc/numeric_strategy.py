from enum import Enum


class NumericStrategy(Enum):
    # (ordinal, gui text, is normal value, is discretiser, is for half interval)
    NUMERIC_BINS = (0, "bins", True, True, True)
    NUMERIC_BEST_BINS = (1, "best-bins", True, True, True)
    NUMERIC_BEST = (2, "best", True, False, True)
    NUMERIC_ALL = (3, "all", True, False, True)
    NUMERIC_INTERVALS = (4, "intervals", False, False, False)  # only valid for SINGLE_NOMINAL
    # XXX MM - temporary code
    # vm = VikaMine style bounded intervals, instead of half intervals
    # consecutive = creates nrBins consecutive equal height intervals
    # all = allow all candidates through, best = only best scoring one
    NUMERIC_VIKAMINE_CONSECUTIVE_ALL = (5, "vm-consecutive-all", True, True, False)
    NUMERIC_VIKAMINE_CONSECUTIVE_BEST = (6, "vm-consecutive-best", True, True, False)

    @property
    def GUI_TEXT(self):
        return self.value[1]

    def __str__(self):
        return self.value[1]

    @staticmethod
    def from_string(text):
        for n in NumericStrategy:
            if n.GUI_TEXT.lower() == (text or "").lower():
                return n
        raise ValueError("NumericStrategy.fromString(): unknown NumericStrategy " + str(text))

    @staticmethod
    def get_normal_values():
        """Everything but NUMERIC_INTERVALS (and the two vm-* ones, they will go)."""
        return [n for n in NumericStrategy if n.value[2] and n not in (
            NumericStrategy.NUMERIC_VIKAMINE_CONSECUTIVE_ALL,
            NumericStrategy.NUMERIC_VIKAMINE_CONSECUTIVE_BEST)]

    def is_discretiser(self):
        return self.value[3]

    def is_for_half_interval(self):
        return self.value[4]
