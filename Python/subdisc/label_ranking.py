import numpy as np

from . import log
from .java_compat import to_f32, fdiv

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def _numeric_value(ch):
    """Character.getNumericValue() for the characters a ranking uses."""
    if '0' <= ch <= '9':
        return ord(ch) - ord('0')
    lo = ch.lower()
    if 'a' <= lo <= 'z':
        return ord(lo) - ord('a') + 10
    return -1


def _signum(x):
    return 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)


class LabelRanking:
    """Labels are numbered 0..k-1, a ranking comes as letters ('a' is label 0),
    e.g. 'b>a>c'. Ties are not implemented properly yet."""

    # static in java, shared by all instances
    _index = []

    def __init__(self, text):
        pref = text.split(">")
        self._size = len(text.replace(">", ""))
        self._ranking = [0] * self._size
        for i, sub in enumerate(pref):
            for ch in sub:
                idx = max(0, _numeric_value(ch) - 10)   # 0 means 'a'
                idx = min(idx, self._size - 1)          # why is this needed?
                self._ranking[idx] = i
        LabelRanking._index = [""] * self._size

    def kendall_tau(self, other):
        numer = np.float32(0.0)
        for i in range(1, self._size):
            for j in range(0, i):
                a = _signum(to_f32(float(self.get_rank(i)) - self.get_rank(j)))
                b = _signum(to_f32(float(other.get_rank(i)) - other.get_rank(j)))
                numer = np.float32(numer + np.float32(a * b))
        return fdiv(to_f32(2 * numer), self._size * (self._size - 1))

    def print(self):
        log.log_command_line("ranking:" + self.get_ranking())

    def get_ranking(self):
        s = ""
        if self.make_index():   # sanity check for robustness
            for i in range(self._size):
                s += LabelRanking._index[i]
                if LabelRanking._index[i] != "" and i < self._size - 1:
                    s += ">"
        else:
            s = "(incorrect ranking target)"
        return s

    def make_index(self):
        LabelRanking._index = [""] * self._size
        for i in range(self._size):
            idx = self._ranking[i]
            # the target column may be malformed
            if 0 <= idx < len(LabelRanking._index):
                LabelRanking._index[idx] += LabelRanking.get_letter(i)
            else:
                log.log_command_line("This target column doesn't seem to be a correct label ranking.")
                return False
        return True

    @staticmethod
    def get_letter(label):
        return ALPHABET[label:label + 1]

    @staticmethod
    def get_label(label):
        return LabelRanking._index[label]

    def get_rank(self, index):
        # not all columns are well formed, this adds some robustness
        if 0 <= index < self._size:
            return self._ranking[index]
        return 0

    def set_rank(self, index, rank):
        self._ranking[index] = rank

    def get_size(self):
        return self._size
