from functools import total_ordering


@total_ordering
class ValueSet:
    """Sorted, unmodifiable set of nominal values (used by 'in' conditions).

    Sorting is plain string order, like the TreeSet<String> in java.
    """

    __slots__ = ("_values", "_lookup")

    def __init__(self, domain):
        values = sorted(set(domain))
        if len(values) == 0:
            raise ValueError("Domains must be > 0")
        self._values = tuple(values)
        self._lookup = frozenset(values)

    def contains(self, value):
        return value in self._lookup

    __contains__ = contains

    def size(self):
        return len(self._values)

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def values(self):
        return self._values

    # NOTE comparing only makes sense for ValueSets of the same Column,
    # the condition comparison assumes that
    def compare_to(self, other):
        """Same size and values -> equal; else smaller sets first; else the
        lexicographically smallest non-shared value decides."""
        if self is other:
            return 0
        cmp = len(self._values) - len(other._values)
        if cmp != 0:
            return cmp
        shared = self._lookup & other._lookup
        if len(self._values) == len(shared):
            return 0
        a = min(v for v in self._values if v not in shared)
        b = min(v for v in other._values if v not in shared)
        return -1 if a < b else (1 if a > b else 0)

    def __eq__(self, other):
        return isinstance(other, ValueSet) and self.compare_to(other) == 0

    def __lt__(self, other):
        return self.compare_to(other) < 0

    def __hash__(self):
        return hash(self._values)

    def __str__(self):
        if not self._values:
            return "{}"
        return "{" + ",".join(self._values) + "}"

    __repr__ = __str__
