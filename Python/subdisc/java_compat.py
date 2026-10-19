"""
Small helpers that mimic the bits of Java behaviour SubDisc depends on.

The Java code works with 32 bit floats almost everywhere, prints numbers
with Float.toString() and draws random numbers from java.util.Random with
fixed seeds. To get the same subgroups and the same output files as the
Java version I copy that behaviour here instead of using the Python
defaults (which print and round a bit differently).
"""

import math
import re
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN, getcontext

import numpy as np

getcontext().prec = 80

f32 = np.float32

FLOAT_MAX = float(np.finfo(np.float32).max)
FLOAT_MIN_VALUE = 1.401298464324817e-45  # Float.MIN_VALUE (smallest subnormal)
INT_MAX = 2147483647
INT_MIN = -2147483648
LONG_MAX = 9223372036854775807
LONG_MIN = -9223372036854775808
DOUBLE_MAX = 1.7976931348623157e308
DOUBLE_MIN_VALUE = 5e-324


def to_f32(x):
    """Round a python number to the nearest float32 and give it back as a python float."""
    return float(np.float32(x))


# ----------------------------------------------------------------------------
# Float.toString / Double.toString
# ----------------------------------------------------------------------------

def _shortest_digits(x, is_float):
    # numpy gives the shortest decimal that round trips, for float32 too
    if is_float:
        s = np.format_float_scientific(np.float32(x), unique=True, trim='-', exp_digits=1)
    else:
        s = np.format_float_scientific(np.float64(x), unique=True, trim='-', exp_digits=1)
    mant, exp = s.split('e')
    mant = mant.lstrip('-')
    digits = mant.replace('.', '')
    return digits, int(exp)


def _rounds_back(dec, x, is_float):
    v = float(dec)
    if is_float:
        return float(np.float32(v)) == x and _f32_from_decimal(dec) == x
    return v == x


def _f32_from_decimal(dec):
    # correctly rounded decimal -> float32 (avoids double rounding)
    d = float(dec)
    f = np.float32(d)
    if float(f) == d or not math.isfinite(d):
        return float(f)
    lo = np.nextafter(f, np.float32(-np.inf)) if float(f) > d else f
    hi = f if float(f) > d else np.nextafter(f, np.float32(np.inf))
    mid = (Decimal(float(lo)) + Decimal(float(hi))) / 2
    if dec < mid:
        return float(lo)
    if dec > mid:
        return float(hi)
    # tie, pick even mantissa
    return float(lo) if (int(np.float32(lo).view(np.int32)) & 1) == 0 else float(hi)


def _java_digits(x, is_float):
    """Digits and exponent as the JDK 19+ algorithm picks them.

    When the shortest decimal has only one digit the JDK also looks at the
    two digit decimals and takes the closest one (e.g. 1.4E-45)."""
    digits, exp = _shortest_digits(x, is_float)
    if len(digits) == 1:
        exact = Decimal(x)
        best = Decimal(digits) * (Decimal(10) ** exp)
        best_dist = abs(best - exact)
        scale = Decimal(10) ** (exp - 1)
        base = int((abs(exact) / scale).to_integral_value(rounding=ROUND_HALF_EVEN))
        for cand in (base - 1, base, base + 1):
            if cand < 10 or cand > 99:
                continue
            dec = Decimal(cand) * scale
            if x < 0:
                dec = -dec
            if not _rounds_back(dec, x, is_float):
                continue
            dist = abs(abs(dec) - abs(exact))
            if dist < best_dist:
                best_dist = dist
                s = str(cand)
                if s[1] == '0':
                    digits, exp = s[0], exp
                else:
                    digits, exp = s, exp
    return digits, exp


def _java_to_string(x, is_float):
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x == 0.0:
        return "-0.0" if math.copysign(1.0, x) < 0 else "0.0"
    sign = "-" if x < 0 else ""
    digits, exp = _java_digits(x, is_float)
    a = abs(x)
    if 1e-3 <= a < 1e7:
        # plain notation
        if exp >= 0:
            if len(digits) <= exp + 1:
                ip = digits + "0" * (exp + 1 - len(digits))
                fp = "0"
            else:
                ip = digits[:exp + 1]
                fp = digits[exp + 1:]
        else:
            ip = "0"
            fp = "0" * (-exp - 1) + digits
        return sign + ip + "." + fp
    # computerized scientific notation
    frac = digits[1:] if len(digits) > 1 else "0"
    return sign + digits[0] + "." + frac + "E" + str(exp)


def float_to_string(x):
    """Java Float.toString(float)."""
    x = float(np.float32(x))
    return _java_to_string(x, True)


def double_to_string(x):
    """Java Double.toString(double)."""
    return _java_to_string(float(x), False)


def jstr(x):
    """String.valueOf() for the types the port uses."""
    if x is None:
        return "null"
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, np.bool_):
        return "true" if bool(x) else "false"
    if isinstance(x, np.float32):
        return float_to_string(x)
    if isinstance(x, (float, np.floating)):
        return double_to_string(x)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def format_fixed(x, precision=6):
    """Java String.format("%.nf"), which rounds HALF_UP on the shortest digits."""
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    d = Decimal(repr(x))
    q = Decimal(1).scaleb(-precision)
    r = d.quantize(q, rounding=ROUND_HALF_UP)
    s = format(r, 'f')
    if s.startswith('-') and float(r) == 0.0 and math.copysign(1.0, x) > 0:
        s = s[1:]
    return s


# ----------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------

_JAVA_FLOAT_RE = re.compile(
    r'^[+-]?(NaN|Infinity|'
    r'((\d+\.?\d*|\.\d+)([eE][+-]?\d+)?)[fFdD]?|'
    r'(0[xX]([0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)[pP][+-]?\d+)[fFdD]?)$')


def jtrim(s):
    """String.trim(): removes every char <= ' ' at both ends."""
    i, j = 0, len(s)
    while i < j and s[i] <= ' ':
        i += 1
    while j > i and s[j - 1] <= ' ':
        j -= 1
    return s[i:j]


def _clean_number(s):
    t = jtrim(s)
    if not _JAVA_FLOAT_RE.match(t):
        raise ValueError("For input string: \"%s\"" % s)
    neg = t.startswith('-')
    body = t.lstrip('+-')
    if body == "NaN":
        return t, float('nan'), None
    if body == "Infinity":
        return t, float('-inf') if neg else float('inf'), None
    if body[-1] in 'fFdD':
        body = body[:-1]
    if body[:2] in ('0x', '0X'):
        v = float.fromhex(body)
        return t, -v if neg else v, None
    dec = Decimal(body)
    if neg:
        dec = -dec
    return t, float(dec), dec


def parse_float(s):
    """Java Float.parseFloat(), result is a python float holding a float32 value."""
    _, v, dec = _clean_number(s)
    if dec is None or not math.isfinite(v):
        return float(np.float32(v))
    if abs(dec) >= Decimal(FLOAT_MAX) * (1 + Decimal(2) ** -25):
        return float('-inf') if dec < 0 else float('inf')
    return _f32_from_decimal(dec)


def parse_double(s):
    """Java Double.parseDouble()."""
    _, v, _ = _clean_number(s)
    return v


def is_parsable_float(s):
    try:
        parse_float(s)
        return True
    except ValueError:
        return False


_JAVA_INT_RE = re.compile(r'^[+-]?\d+$')


def parse_int(s):
    """Java Integer.parseInt(), strict: no spaces, int32 range."""
    if s is None or not _JAVA_INT_RE.match(s):
        raise ValueError("For input string: \"%s\"" % s)
    v = int(s)
    if v < INT_MIN or v > INT_MAX:
        raise ValueError("For input string: \"%s\"" % s)
    return v


def parse_boolean(s):
    """Java Boolean.parseBoolean()."""
    return s is not None and s.lower() == "true"


def java_split(s, regex, limit=0):
    """String.split(regex, limit) with the Java rules for empty strings."""
    if s == "":
        return [""]
    if limit > 0:
        parts = re.split(regex, s, maxsplit=limit - 1)
    else:
        parts = re.split(regex, s)
    # java 8+: a zero width match at index 0 never gives a leading empty string
    if limit == 0:
        while len(parts) > 1 and parts[-1] == "":
            parts.pop()
        if len(parts) == 1 and parts[0] == "" :
            return [""]
    return parts


def java_string_hash(s):
    h = 0
    for ch in s:
        h = (31 * h + ord(ch)) & 0xFFFFFFFF
    return h - (1 << 32) if h >= (1 << 31) else h


def float_compare(a, b):
    """Float.compare(): NaN equals NaN and is the biggest, -0.0 < 0.0."""
    a_nan, b_nan = math.isnan(a), math.isnan(b)
    if a_nan or b_nan:
        if a_nan and b_nan:
            return 0
        return 1 if a_nan else -1
    if a < b:
        return -1
    if a > b:
        return 1
    sa, sb = math.copysign(1.0, a), math.copysign(1.0, b)
    if sa == sb:
        return 0
    return -1 if sa < sb else 1


def java_sort_floats(arr):
    """Arrays.sort(float[]): ascending, -0.0 before 0.0, NaN at the end."""
    a = np.asarray(arr, dtype=np.float32)
    if a.size == 0:
        return a.copy()
    out = np.sort(a, kind='stable')
    # numpy keeps -0.0 / 0.0 in input order, fix them up like java does
    zeros = np.flatnonzero(out == 0)
    if zeros.size > 1:
        neg = np.signbit(out[zeros])
        n_neg = int(np.count_nonzero(neg))
        out[zeros[:n_neg]] = np.float32(-0.0)
        out[zeros[n_neg:]] = np.float32(0.0)
    return out


def binary_search(sorted_arr, key):
    """Arrays.binarySearch(float[], float) including the -(insertion)-1 convention."""
    lo, hi = 0, len(sorted_arr) - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        c = float_compare(float(sorted_arr[mid]), float(key))
        if c < 0:
            lo = mid + 1
        elif c > 0:
            hi = mid - 1
        else:
            return mid
    return -(lo + 1)


def int32(x):
    """Wrap a python int like a java int overflow would."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


def int64(x):
    x &= 0xFFFFFFFFFFFFFFFF
    return x - (1 << 64) if x >= (1 << 63) else x


def java_round(x):
    """Math.round(double) -> long, rounds half up."""
    if math.isnan(x):
        return 0
    return int(math.floor(x + 0.5))


def java_idiv(a, b):
    """Integer division truncating towards zero, like java."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def java_imod(a, b):
    return a - b * java_idiv(a, b)


# ----------------------------------------------------------------------------
# java.lang.Math look-alikes: never raise, give inf / nan like java does
# ----------------------------------------------------------------------------

NaN = float('nan')
POS_INF = float('inf')
NEG_INF = float('-inf')


def div(a, b):
    """a / b for doubles with IEEE results instead of ZeroDivisionError."""
    a = float(a)
    b = float(b)
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return NaN
        neg = (a < 0) != (math.copysign(1.0, b) < 0)
        return NEG_INF if neg else POS_INF
    return a / b


def fdiv(a, b):
    """float / float in single precision (java float arithmetic)."""
    a = np.float32(a)
    b = np.float32(b)
    with np.errstate(all='ignore'):
        return float(a / b)


def fmul(a, b):
    with np.errstate(all='ignore'):
        return float(np.float32(a) * np.float32(b))


def fadd(a, b):
    with np.errstate(all='ignore'):
        return float(np.float32(a) + np.float32(b))


def fsub(a, b):
    with np.errstate(all='ignore'):
        return float(np.float32(a) - np.float32(b))


def jlog(x):
    x = float(x)
    if math.isnan(x) or x < 0:
        return NaN
    if x == 0:
        return NEG_INF
    if math.isinf(x):
        return POS_INF
    return math.log(x)


def jlog10(x):
    x = float(x)
    if math.isnan(x) or x < 0:
        return NaN
    if x == 0:
        return NEG_INF
    if math.isinf(x):
        return POS_INF
    return math.log10(x)


def jsqrt(x):
    x = float(x)
    if math.isnan(x) or x < 0:
        return NaN
    if math.isinf(x):
        return POS_INF
    return math.sqrt(x)


def jexp(x):
    x = float(x)
    try:
        return math.exp(x)
    except OverflowError:
        return POS_INF


def jpow(a, b):
    """Math.pow() edge cases without exceptions."""
    a = float(a)
    b = float(b)
    try:
        r = math.pow(a, b)
        return r
    except OverflowError:
        if a < 0 and b == int(b) and int(b) % 2 == 1:
            return NEG_INF
        return POS_INF
    except (ValueError, ZeroDivisionError):
        if a == 0.0 and b < 0:
            if math.copysign(1.0, a) < 0 and b == int(b) and int(b) % 2 == 1:
                return NEG_INF
            return POS_INF
        return NaN


def jabs(x):
    return abs(x)


def jsignum(x):
    if math.isnan(x):
        return NaN
    if x > 0:
        return 1.0
    if x < 0:
        return -1.0
    return x


def jmax(a, b):
    """Math.max for doubles: NaN wins."""
    if math.isnan(a) or math.isnan(b):
        return NaN
    return a if a >= b else b


def jmin(a, b):
    if math.isnan(a) or math.isnan(b):
        return NaN
    return a if a <= b else b


def to_int(x):
    """(int) cast of a double: truncates, NaN -> 0, clamps at int range."""
    if math.isnan(x):
        return 0
    if x >= INT_MAX:
        return INT_MAX
    if x <= INT_MIN:
        return INT_MIN
    return int(x)


def to_long(x):
    if math.isnan(x):
        return 0
    if x >= LONG_MAX:
        return LONG_MAX
    if x <= LONG_MIN:
        return LONG_MIN
    return int(x)


# ----------------------------------------------------------------------------
# java.util.Random
# ----------------------------------------------------------------------------

class JavaRandom:
    """Same sequence as java.util.Random for the same seed."""

    _MULT = 0x5DEECE66D
    _ADD = 0xB
    _MASK = (1 << 48) - 1

    def __init__(self, seed=None):
        if seed is None:
            import time
            import os
            seed = int.from_bytes(os.urandom(8), 'little') ^ time.perf_counter_ns()
        self.set_seed(seed)

    def set_seed(self, seed):
        self._seed = (int(seed) ^ self._MULT) & self._MASK
        self._next_next_gaussian = None

    def _next(self, bits):
        self._seed = (self._seed * self._MULT + self._ADD) & self._MASK
        return int32(self._seed >> (48 - bits))

    def next_int(self, bound=None):
        if bound is None:
            return self._next(32)
        if bound <= 0:
            raise ValueError("bound must be positive")
        if (bound & -bound) == bound:
            return int32((bound * self._next(31)) >> 31)
        while True:
            bits = self._next(31)
            val = bits % bound
            if bits - val + (bound - 1) < (1 << 31):
                return val

    def next_long(self):
        return int64((self._next(32) << 32) + self._next(32))

    def next_boolean(self):
        return self._next(1) != 0

    def next_float(self):
        return self._next(24) / float(1 << 24)

    def next_double(self):
        return ((self._next(26) << 27) + self._next(27)) * (1.0 / (1 << 53))

    def next_gaussian(self):
        if self._next_next_gaussian is not None:
            g = self._next_next_gaussian
            self._next_next_gaussian = None
            return g
        while True:
            v1 = 2 * self.next_double() - 1
            v2 = 2 * self.next_double() - 1
            s = v1 * v1 + v2 * v2
            if 0 < s < 1:
                break
        # StrictMath.log / sqrt, plain math is close enough here
        multiplier = math.sqrt(-2 * math.log(s) / s)
        self._next_next_gaussian = v2 * multiplier
        return v1 * multiplier


def java_shuffle(lst, rnd):
    """Collections.shuffle(list, rnd) for RandomAccess lists."""
    for i in range(len(lst) - 1, 0, -1):
        j = rnd.next_int(i + 1)
        lst[i], lst[j] = lst[j], lst[i]
