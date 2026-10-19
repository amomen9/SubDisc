"""A small sorted set with java.util.TreeSet semantics.

Elements are ordered by a compare function (cmp(a, b) -> int). An element
that compares equal (0) to one already present is not added, exactly like
a TreeSet. Storage is a list of short sorted buckets so inserts stay cheap
for big candidate queues (same idea as the sortedcontainers package).
"""

from bisect import bisect_left, bisect_right
from functools import cmp_to_key

_LOAD = 512


class TreeSet:
    __slots__ = ("_cmp", "_key", "_buckets", "_maxes", "_len")

    def __init__(self, cmp=None, items=None):
        if cmp is None:
            cmp = _natural
        self._cmp = cmp
        self._key = cmp_to_key(cmp)
        self._buckets = []   # list of lists of (key, item)
        self._maxes = []     # key of the last element of every bucket
        self._len = 0
        if items is not None:
            for it in items:
                self.add(it)

    def comparator(self):
        return self._cmp

    def __len__(self):
        return self._len

    def size(self):
        return self._len

    def is_empty(self):
        return self._len == 0

    def clear(self):
        self._buckets = []
        self._maxes = []
        self._len = 0

    def _locate(self, k):
        """bucket index and position of the first element >= k"""
        maxes = self._maxes
        if not maxes:
            return 0, 0
        b = bisect_left(maxes, k)
        if b == len(maxes):
            b -= 1
            bucket = self._buckets[b]
            return b, len(bucket)
        bucket = self._buckets[b]
        keys = [e[0] for e in bucket] if len(bucket) < 32 else None
        if keys is not None:
            return b, bisect_left(keys, k)
        lo, hi = 0, len(bucket)
        while lo < hi:
            mid = (lo + hi) >> 1
            if bucket[mid][0] < k:
                lo = mid + 1
            else:
                hi = mid
        return b, lo

    def add(self, item):
        """True when added, False when an equal element is present."""
        k = self._key(item)
        if not self._buckets:
            self._buckets.append([(k, item)])
            self._maxes.append(k)
            self._len = 1
            return True
        b, pos = self._locate(k)
        bucket = self._buckets[b]
        if pos < len(bucket) and self._cmp(bucket[pos][1], item) == 0:
            return False
        bucket.insert(pos, (k, item))
        if pos == len(bucket) - 1:
            self._maxes[b] = k
        self._len += 1
        if len(bucket) > 2 * _LOAD:
            half = len(bucket) // 2
            self._buckets.insert(b + 1, bucket[half:])
            del bucket[half:]
            self._maxes.insert(b + 1, self._buckets[b + 1][-1][0])
            self._maxes[b] = bucket[-1][0]
        return True

    def contains(self, item):
        if not self._buckets:
            return False
        k = self._key(item)
        b, pos = self._locate(k)
        bucket = self._buckets[b]
        return pos < len(bucket) and self._cmp(bucket[pos][1], item) == 0

    __contains__ = contains

    def remove(self, item):
        """removes the element that compares equal to item"""
        if not self._buckets:
            return False
        k = self._key(item)
        b, pos = self._locate(k)
        bucket = self._buckets[b]
        if pos < len(bucket) and self._cmp(bucket[pos][1], item) == 0:
            self._delete(b, pos)
            return True
        return False

    discard = remove

    def _delete(self, b, pos):
        bucket = self._buckets[b]
        del bucket[pos]
        self._len -= 1
        if not bucket:
            del self._buckets[b]
            del self._maxes[b]
        else:
            self._maxes[b] = bucket[-1][0]

    def first(self):
        if not self._len:
            raise KeyError("NoSuchElementException")
        return self._buckets[0][0][1]

    def last(self):
        if not self._len:
            raise KeyError("NoSuchElementException")
        return self._buckets[-1][-1][1]

    def poll_first(self):
        """removes and returns the first element, None when empty"""
        if not self._len:
            return None
        item = self._buckets[0][0][1]
        self._delete(0, 0)
        return item

    def poll_last(self):
        if not self._len:
            return None
        b = len(self._buckets) - 1
        item = self._buckets[b][-1][1]
        self._delete(b, len(self._buckets[b]) - 1)
        return item

    def __iter__(self):
        for bucket in list(self._buckets):
            for e in list(bucket):
                yield e[1]

    def to_list(self):
        return [e[1] for bucket in self._buckets for e in bucket]

    def get(self, index):
        """index access, O(number of buckets)"""
        if index < 0:
            index += self._len
        for bucket in self._buckets:
            if index < len(bucket):
                return bucket[index][1]
            index -= len(bucket)
        raise IndexError(index)

    def add_all(self, items):
        changed = False
        for it in items:
            changed |= self.add(it)
        return changed

    def remove_if(self, predicate):
        keep = [it for it in self.to_list() if not predicate(it)]
        removed = self._len - len(keep)
        if removed:
            self.clear()
            for it in keep:
                self._append_sorted(it)
        return removed

    def _append_sorted(self, item):
        # caller guarantees item goes at the end
        k = self._key(item)
        if not self._buckets or len(self._buckets[-1]) >= _LOAD:
            self._buckets.append([])
            self._maxes.append(k)
        self._buckets[-1].append((k, item))
        self._maxes[-1] = k
        self._len += 1


def _natural(a, b):
    return a.compare_to(b)
