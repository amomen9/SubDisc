"""ROCList: the SubgroupROCPoints that form the convex hull, sorted on FPR."""

from .java_compat import div
from .roc_point import SubgroupROCPoint


class ROCList(list):
    def __init__(self, subgroup_set=None):
        super().__init__()
        if subgroup_set is not None:
            for s in subgroup_set:
                self.add(SubgroupROCPoint(s))

    def add(self, p):
        index = len(self)
        tpr = p.get_tpr()
        fpr = p.get_fpr()
        # always under the curve
        if tpr < fpr:
            return False
        self.append(p)
        if len(self) > 1:
            # move the new rule from the end to its place
            while index > 0 and self._fpr_at(index - 1) >= fpr:
                other = self[index - 1]
                self[index - 1] = p
                self[index] = other
                index -= 1
            if self._fpr_at(index + 1) == self._fpr_at(index):
                if self._tpr_at(index + 1) > self._tpr_at(index):
                    del self[index]
                else:
                    del self[index + 1]
            if self._slope_at(index - 1) <= self._slope_at(index):
                del self[index]
            while index > 0 and self._slope_at(index - 2) <= self._slope_at(index - 1):
                del self[index - 1]
                index -= 1
            while index < len(self) - 1 and self._slope_at(index) <= self._slope_at(index + 1):
                del self[index + 1]
        return True

    def _tpr_at(self, i):
        if i == -1:
            return 0.0
        if i == len(self):
            return 1.0
        return self[i].get_tpr()

    def _fpr_at(self, i):
        if i == -1:
            return 0.0
        if i == len(self):
            return 1.0
        return self[i].get_fpr()

    def _slope_at(self, i):
        n = len(self)
        if n == 0:
            return 0.0
        if i == -1:
            return div(self._tpr_at(0), self._fpr_at(0))
        if i == n:
            return 0.0
        if i == n - 1:
            return div(1.0 - self._tpr_at(i), 1.0 - self._fpr_at(i))
        if self._fpr_at(i + 1) == self._fpr_at(i):
            return 0.0
        return div(self._tpr_at(i + 1) - self._tpr_at(i), self._fpr_at(i + 1) - self._fpr_at(i))

    def get_area_under_curve(self):
        area = 0.0
        for i in range(-1, len(self)):
            width = self._fpr_at(i + 1) - self._fpr_at(i)
            area += width * ((self._tpr_at(i) + self._tpr_at(i + 1)) / 2.0)
        return area
