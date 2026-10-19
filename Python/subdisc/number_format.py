"""NumberFormat.getNumberInstance() with at most 6 fraction digits, as the
java result tables show numbers (grouping ',' and decimal point '.')."""

import math
from decimal import Decimal, ROUND_HALF_EVEN


def format_number(x, max_fraction_digits=6, grouping=True):
    x = float(x)
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "∞" if x > 0 else "-∞"
    d = Decimal(x).quantize(Decimal(1).scaleb(-max_fraction_digits), rounding=ROUND_HALF_EVEN)
    neg = d < 0
    d = abs(d)
    s = format(d, 'f')
    if '.' in s:
        ip, fp = s.split('.')
        fp = fp.rstrip('0')
    else:
        ip, fp = s, ''
    if grouping:
        ip = "{:,}".format(int(ip))
    out = ip + ('.' + fp if fp else '')
    if neg and out.strip('0.,') != '':
        out = '-' + out
    return out
