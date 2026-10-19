"""A set of column indices (java ItemSet extends BitSet), kept as an int bitmask."""


class ItemSet:
    __slots__ = ("_bits", "_dimensions", "_joint_entropy")

    def __init__(self, dimensions, count=0):
        """first 'count' items set (all when count > dimensions)"""
        self._dimensions = dimensions
        n = min(count, dimensions)
        self._bits = (1 << n) - 1 if n > 0 else 0
        self._joint_entropy = float('nan')

    def get(self, i):
        return (self._bits >> i) & 1 == 1

    def set(self, i, value=True):
        if value:
            self._bits |= (1 << i)
        else:
            self._bits &= ~(1 << i)

    def clear(self, i=None):
        if i is None:
            self._bits = 0
        else:
            self._bits &= ~(1 << i)

    def bits(self):
        return self._bits

    def symmetric_difference(self, other):
        s = ItemSet(self._dimensions)
        mask = (1 << self._dimensions) - 1
        s._bits = (self._bits ^ other._bits) & mask
        return s

    def get_item_count(self):
        mask = (1 << self._dimensions) - 1
        return bin(self._bits & mask).count("1")

    def cardinality(self):
        return bin(self._bits).count("1")

    def get_dimensions(self):
        return self._dimensions

    def get_extension(self, index):
        s = self.clone()
        s.set(index)
        return s

    def clone(self):
        s = ItemSet(self._dimensions)
        s._bits = self._bits
        s._joint_entropy = self._joint_entropy
        return s

    def set_bits(self):
        """indices of the set items, ascending"""
        out = []
        b = self._bits
        i = 0
        while b:
            if b & 1:
                out.append(i)
            b >>= 1
            i += 1
        return out

    def get_joint_entropy(self):
        return self._joint_entropy

    def set_joint_entropy(self, e):
        self._joint_entropy = e

    def __str__(self):
        return "{" + ", ".join(str(i) for i in self.set_bits()) + "}"

    __repr__ = __str__

    def __eq__(self, other):
        return isinstance(other, ItemSet) and self._bits == other._bits

    def __hash__(self):
        return hash(self._bits)
