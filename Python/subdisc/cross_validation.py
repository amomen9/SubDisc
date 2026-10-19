import time

import numpy as np

from .java_compat import JavaRandom


class CrossValidation:
    """k random folds over the rows. NOTE the random is static and seeded with
    the clock in java, so folds differ between runs."""

    _random = None

    def __init__(self, size, k):
        self._size = size
        self._k = k
        CrossValidation._random = JavaRandom(int(time.time() * 1000))
        self._sets = None
        self.create_test_sets()

    def get_random_permutation(self):
        """random permutation of 1..size (Knuth shuffle)"""
        result = list(range(1, self._size + 1))
        rnd = CrossValidation._random
        for i in range(self._size, 1, -1):
            j = rnd.next_int(i)
            result[j], result[i - 1] = result[i - 1], result[j]
        return result

    def create_test_sets(self):
        perm = self.get_random_permutation()
        sets = [0] * self._size
        for i in range(self._size):
            sets[perm[i] - 1] = i % self._k
        self._sets = sets

    def is_in_test_set(self, index, test_set):
        return self._sets[index] == test_set

    def get_test_set_number(self, index):
        return self._sets[index]

    def get_set(self, test_set, invert):
        """test set (invert False) or the training set (invert True)"""
        s = np.array(self._sets, dtype=np.int64)
        return (s == test_set) != invert
