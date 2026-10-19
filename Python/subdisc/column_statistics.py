"""Statistics of a subgroup (and its complement) for a numeric target.

The java class works with floats (and says so in a FIXME), I keep the
float32 arithmetic so the quality values come out the same.
"""

import math

from . import log
from .java_compat import fdiv, fadd, to_f32, jsqrt, div

NAN = float('nan')


class Statistics:
    def __init__(self, coverage=0, subgroup_sum=NAN, ssd=NAN, median=NAN, mad=NAN):
        self._coverage = coverage
        self._complement_coverage = 0
        self._has_complement = False
        self._subgroup_sum = to_f32(subgroup_sum)
        self._subgroup_ssd = to_f32(ssd)
        self._median = to_f32(median)
        self._mad = to_f32(mad)
        # complement
        self._complement_sum = NAN
        self._complement_ssd = NAN
        # whole data, coverage and sum follow from subgroup + complement
        self._ssd = NAN

    def add_complement(self, complement_coverage, complement_sum, complement_ssd):
        self._has_complement = True
        self._complement_coverage = complement_coverage
        self._complement_sum = to_f32(complement_sum)
        self._complement_ssd = to_f32(complement_ssd)

    def add_dataset_ssd(self, ssd):
        self._ssd = to_f32(ssd)

    def get_coverage(self):
        return self._coverage

    def get_subgroup_sum(self):
        return self._subgroup_sum

    def get_subgroup_average(self):
        return fdiv(self._subgroup_sum, self._coverage)

    def get_subgroup_sum_squared_deviations(self):
        return self._subgroup_ssd

    def get_median(self):
        return self._median

    def get_median_absolute_deviations(self):
        return self._mad

    # complement
    def get_complement_coverage(self):
        return self._complement_coverage

    def get_complement_sum(self):
        return self._complement_sum

    def get_complement_average(self):
        return fdiv(self._complement_sum, self._complement_coverage)

    def get_complement_sum_squared_deviations(self):
        return self._complement_ssd

    def get_complement_standard_deviation(self):
        return to_f32(jsqrt(div(self._complement_ssd, float(self._complement_coverage))))

    def get_sum_squared_deviations(self):
        return self._ssd

    def get_average(self):
        """Average of all data (may be just a selection of the data)."""
        if self._has_complement:
            return fdiv(fadd(self._subgroup_sum, self._complement_sum), self._coverage + self._complement_coverage)
        return fdiv(self._subgroup_sum, self._coverage)

    def print(self):
        t = "total: %f\n" % self.get_average()
        s = "subgroup (%d): %f, %f\n" % (self.get_coverage(), self.get_subgroup_average(),
                                         jsqrt(div(self._subgroup_ssd, float(self._coverage))))
        c = "complement (%d): %f, %f\n" % (self.get_complement_coverage(), self.get_complement_average(),
                                           jsqrt(div(self._complement_ssd, float(self._complement_coverage))))
        log.log_command_line(t + s + c)
