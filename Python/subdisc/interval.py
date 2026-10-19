import math
from functools import total_ordering

from .java_compat import float_compare, float_to_string, to_f32


@total_ordering
class Interval:
    """Half open interval (lower, upper]."""

    __slots__ = ("_lower", "_upper")

    def __init__(self, lower, upper):
        lower = to_f32(lower)
        upper = to_f32(upper)
        if math.isnan(lower):
            raise ValueError("Interval<init>: theLower can not be NaN.")
        if math.isnan(upper):
            raise ValueError("Interval<init>: theUpper can not be NaN.")
        # -0.0 < 0.0, NaN dealt with above
        if float_compare(lower, upper) > 0:
            raise ValueError("Interval<init>: the theLower must be <= theUpper.")
        self._lower = lower
        self._upper = upper

    def get_lower(self):
        return self._lower

    def get_upper(self):
        return self._upper

    def between(self, value):
        return self._lower < value <= self._upper

    def between_mask(self, values):
        """Vectorised between() for a numpy array."""
        return (values > self._lower) & (values <= self._upper)

    def __str__(self):
        left = "-inf" if self._lower == float('-inf') else float_to_string(self._lower)
        right = "inf)" if self._upper == float('inf') else (float_to_string(self._upper) + "]")
        return "(" + left + ", " + right

    __repr__ = __str__

    def compare_to(self, other):
        if self is other:
            return 0
        # NOTE considers 0.0 to be greater than -0.0
        cmp = float_compare(self._lower, other._lower)
        return cmp if cmp != 0 else float_compare(self._upper, other._upper)

    def __eq__(self, other):
        return isinstance(other, Interval) and self.compare_to(other) == 0

    def __lt__(self, other):
        return self.compare_to(other) < 0

    def __hash__(self):
        return hash((self._lower, self._upper))
