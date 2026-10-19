"""Points: PointDouble, PointFloat, ROCPoint, SubgroupROCPoint, CandidateROCPoint."""

import math

from . import log
from .java_compat import double_to_string, float_to_string, to_f32


class PointDouble:
    """Like Point2D.Double, but immutable; NaN and infinity are not allowed."""
    __slots__ = ("x", "y")

    def __init__(self, x, y):
        x = float(x)
        y = float(y)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError("NaN and (+/-)Infinity are not allowed")
        self.x = x
        self.y = y

    def print(self):
        log.log_command_line(str(self))

    def __str__(self):
        return "PointDouble[" + double_to_string(self.x) + ", " + double_to_string(self.y) + "]"


class PointFloat:
    __slots__ = ("x", "y")

    def __init__(self, x, y=None):
        if y is None and isinstance(x, PointFloat):
            x, y = x.x, x.y
        self.x = to_f32(x)
        self.y = to_f32(y)

    def __str__(self):
        return "PointFloat[" + float_to_string(self.x) + ", " + float_to_string(self.y) + "]"

    def print(self):
        log.log_command_line(str(self))


class ROCPoint(PointDouble):
    """Both coordinates must lie in [0, 1]."""
    __slots__ = ()

    def __init__(self, x, y):
        PointDouble.__init__(self, x, y)
        if not (0.0 <= self.x <= 1.0) or not (0.0 <= self.y <= 1.0):
            raise ValueError("Coordinates must lie in interval [0-1] in both dimensions")

    def get_fpr(self):
        return self.x

    def get_tpr(self):
        return self.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, ROCPoint):
            return self.x == other.x and self.y == other.y
        return self is other

    def __str__(self):
        return "ROCPoint[" + double_to_string(self.x) + ", " + double_to_string(self.y) + "]"


class SubgroupROCPoint(ROCPoint):
    """ROCPoint of a Subgroup: ID, FPR as x, TPR as y.
    FIXME MM (java) the ID may change during the life of a Subgroup."""
    __slots__ = ("ID", "_subgroup")

    def __init__(self, subgroup):
        ROCPoint.__init__(self, subgroup.get_false_positive_rate(), subgroup.get_true_positive_rate())
        self._subgroup = subgroup
        self.ID = subgroup.get_id()

    def get_subgroup(self):
        return self._subgroup

    def __str__(self):
        return "Subgroup " + str(self.ID) + " (FPR " + double_to_string(self.x) + ", TPR " + double_to_string(self.y) + ")"

    # identity, like the java points in hull lists
    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other


class CandidateROCPoint(SubgroupROCPoint):
    __slots__ = ("_candidate", "slope")

    def __init__(self, candidate):
        SubgroupROCPoint.__init__(self, candidate.get_subgroup())
        self._candidate = candidate
        self.slope = float('nan')

    def get_candidate(self):
        return self._candidate

    def __str__(self):
        return "CandidateROCPoint(" + double_to_string(self.x) + " ," + double_to_string(self.y) + ") " + str(self._candidate)
