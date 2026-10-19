"""Normal distribution helpers.

NOTE the java fields are static, so every new NormalDistribution changes mu
and sigma for all of them. Kept that way (class attributes) on purpose,
results of the validation code depend on it.
"""

import math
import time

from . import log
from .java_compat import JavaRandom, jsqrt, jexp, jpow, div, to_f32, double_to_string, float_to_string


class NormalDistribution:
    _mu = 0.0
    _sigma = 1.0
    _random = None

    def __init__(self, mean_or_sample=None, variance=None):
        if mean_or_sample is None:
            # standard normal distribution
            NormalDistribution._mu = 0.0
            NormalDistribution._sigma = 1.0
        elif variance is not None:
            NormalDistribution._mu = float(mean_or_sample)
            NormalDistribution._sigma = jsqrt(variance)
        else:
            sample = list(mean_or_sample)
            total = 0.0
            for v in sample:
                total += v
            mu = div(total, len(sample))
            sq = 0.0
            for v in sample:
                d = v - mu
                sq += d * d
            NormalDistribution._mu = mu
            NormalDistribution._sigma = jsqrt(div(sq, len(sample) - 1))
        NormalDistribution._random = JavaRandom(int(time.time() * 1000))

    def get_next_double(self):
        return NormalDistribution._sigma * NormalDistribution._random.next_gaussian() + NormalDistribution._mu

    def calc_pdf(self, x):
        mu, s = NormalDistribution._mu, NormalDistribution._sigma
        return div(jpow(math.e, div(-jpow(x - mu, 2), 2 * s * s)), s * math.sqrt(2 * math.pi))

    def calc_cdf(self, x):
        mu, s = NormalDistribution._mu, NormalDistribution._sigma
        return 0.5 * (1 + self.calc_erf(div(x - mu, s * math.sqrt(2))))

    def calc_erf(self, z):
        """Horner's method, fractional error below 1.2e-7"""
        t = 1.0 / (1.0 + 0.5 * abs(z))
        r = 1 - t * jexp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
                     + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
        return r if z >= 0 else -r

    def z_transform(self, value):
        return div(value - NormalDistribution._mu, NormalDistribution._sigma)

    def get_mu(self):
        return NormalDistribution._mu

    def get_sigma(self):
        return NormalDistribution._sigma

    def get_variance(self):
        return NormalDistribution._sigma * NormalDistribution._sigma

    def get_one_percent_significance(self):
        return to_f32(NormalDistribution._mu + 2.326348 * NormalDistribution._sigma)

    def get_five_percent_significance(self):
        return to_f32(NormalDistribution._mu + 1.6448537 * NormalDistribution._sigma)

    def get_ten_percent_significance(self):
        return to_f32(NormalDistribution._mu + 1.2815517 * NormalDistribution._sigma)

    def print_statistics(self):
        log.log_command_line("====================================")
        log.log_command_line("Overall statistics:")
        log.log_command_line("  mu    = " + double_to_string(NormalDistribution._mu))
        log.log_command_line("  sigma = " + double_to_string(NormalDistribution._sigma))
        log.log_command_line("------------------------------------")
        log.log_command_line("Significant quality cutoff values:")
        log.log_command_line("  alpha =  1% : " + float_to_string(self.get_one_percent_significance()))
        log.log_command_line("  alpha =  5% : " + float_to_string(self.get_five_percent_significance()))
        log.log_command_line("  alpha = 10% : " + float_to_string(self.get_ten_percent_significance()))
