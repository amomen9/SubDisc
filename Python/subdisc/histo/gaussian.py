"""Gaussian helpers: pdf phi, cdf Phi (Taylor series), inverse by bisection."""

import math

from . import vec

DEBUG = False
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def phi(x, mu=None, sigma=None):
    """standard normal pdf, or N(mu, sigma^2) pdf when mu and sigma are given"""
    if mu is not None:
        return phi((x - mu) / sigma) / sigma
    try:
        return math.exp(-x * x / 2.0) / _SQRT_2PI
    except OverflowError:
        return 0.0


def Phi(z, mu=None, sigma=None):
    """standard normal cdf (Taylor approximation)"""
    if mu is not None:
        return Phi((z - mu) / sigma)
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0
    s = 0.0
    term = z
    i = 3
    while s + term != s:
        s = s + term
        term = term * z * z / i
        i += 2
    return 0.5 + s * phi(z)


def Phi_inverse(y):
    if y < 0.0 or y > 1.0:
        raise ValueError("0 <= y <= 1 violated: " + str(y))
    if y == 0.0:
        return -8.0
    if y == 0.5:
        return 0.0
    if y == 1.0:
        return 8.0
    lo, hi, delta = -8.0, 8.0, 0.00000001
    while True:
        mid = lo + ((hi - lo) / 2.0)
        if hi - lo < delta:
            return mid
        if Phi(mid) > y:
            hi = mid
        else:
            lo = mid


class Gaussian:
    """normal distribution N(mu, sigma^2) sampled at 'samples' points"""

    def __init__(self, mu, sigma, samples):
        if sigma <= 0.0:
            raise ValueError("sigma must be > 0.0")
        if samples < 0 or (samples & 1) == 0:
            raise ValueError("samples must be odd and positive")
        half = (samples + 1) // 2
        p = vec.range_(0.5, 0.5 / (half - 1), half)
        p[-1] = 1.0   # adjust for possible rounding errors
        z = _z(p)
        self._mu = mu
        self._sigma = sigma
        self._x = vec.mirror(z)
        self._fx = _n(z)
        if mu != 0.0 or sigma != 1.0:
            self._x = [mu + v * sigma for v in self._x]
        if sigma != 1.0:
            vec.divide(self._fx, sigma)

    def mu(self):
        return self._mu

    def sigma(self):
        return self._sigma

    def x(self):
        return self._x

    def fx(self):
        return self._fx

    def apply_to(self, data):
        return [[k * d for k in self._fx] for d in data]


def g(sigma, samples):
    out = n(samples)
    vec.divide(out, sigma)
    return out


def n(samples):
    half = (samples + 1) // 2
    p = vec.range_(0.5, 0.5 / (half - 1), half)
    return _n(_z(p))


def _n(z):
    size = 2 * len(z) - 1
    out = [0.0] * size
    mid = size // 2
    for zi in range(len(z)):
        out[mid - zi] = out[mid + zi] = phi(z[zi])
    return out


def _z(p):
    return [Phi_inverse(v) for v in p]
