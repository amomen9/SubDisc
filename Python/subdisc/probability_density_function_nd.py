"""Multi-dimensional kernel density estimates, for the MULTI_NUMERIC target.

Only 2D is implemented for the grid based measures (like java), L2 works
for any dimension. Sums run in the java order (np.cumsum is sequential),
the exp() evaluations are vectorised per grid row.
"""

import math

import numpy as np

from .java_compat import double_to_string, float_to_string, div, jsqrt
from .qm import QM
from .target_type import TargetType

CUTOFF = 3.0
SAMPLES = 35
GRID_STATS = 3
MIN = 0
MAX = 1
GRID_SIZE = 2
X_RESOLUTION = math.sqrt(2.0)
Y_RESOLUTION = math.sqrt(2.0)

MEAN_D1, MEAN_D2, VAR_D1, VAR_D2, COV_1_2, H, SIZE_N, NR_STATS = range(8)

DEBUG = True
F = np.float32


def _debug(msg):
    if DEBUG:
        print(msg)


def _dstr(a):
    return "[" + ", ".join(double_to_string(v) for v in a) + "]"


def _fstr(a):
    return "[" + ", ".join(float_to_string(v) for v in a) + "]"


def _last_cumsum(values, start=0.0):
    """start + v0 + v1 + ... added one by one"""
    if values.size == 0:
        return start
    return float(np.cumsum(np.concatenate(([start], values.ravel())))[-1])


class ProbabilityDensityFunctionND:
    def __init__(self, columns, cutoff=CUTOFF, nr_samples=SAMPLES):
        self._data = _get_data(columns)            # n x d, float32
        self._grid = _get_grid(columns, cutoff, nr_samples)
        self._limits = [[c.get_min(), c.get_max()] for c in columns]
        self.last_dxdy = None

    def get_limits(self):
        return [list(l) for l in self._limits]

    def get_nr_dimensions(self):
        return len(self._grid)

    # ------------------------------------------------------------------

    def get_density(self, members):
        """2D density on the fixed grid"""
        x_min, x_max, x_n = self._grid[0][MIN], self._grid[0][MAX], int(self._grid[0][GRID_SIZE])
        y_min, y_max, y_n = self._grid[1][MIN], self._grid[1][MAX], int(self._grid[1][GRID_SIZE])
        dx = (x_max - x_min) / x_n
        dy = (y_max - y_min) / y_n
        mu = _get_mu_vector(self._data, members)
        cm = _get_covariance_matrix(self._data, members, mu)
        cm_inv = _inverse(cm)
        x_sd = jsqrt(float(cm[0][0]))
        y_sd = jsqrt(float(cm[1][1]))
        x_cut = CUTOFF * x_sd
        y_cut = CUTOFF * y_sd
        n = int(np.count_nonzero(members))
        # det(cm) * n is float * int -> float in java
        f = float(F(1.0 / (2.0 * math.pi * jsqrt(float(F(_det(cm) * F(n)))))))
        xv = float(cm_inv[0][0])
        cov2 = float(cm_inv[0][1]) * 2.0
        yv = float(cm_inv[1][1])
        pts = self._data[np.asarray(members[:self._data.shape[0]], dtype=bool)].astype(np.float64)
        dens = np.zeros((x_n, y_n), dtype=np.float32)
        ys = y_min + (np.arange(y_n) * dy)
        for i in range(x_n):
            if i % 1000 == 0:
                print("grid row: " + str(i))
            x = x_min + (i * dx)
            px = x - pts[:, 0]
            py = ys[:, None] - pts[None, :, 1]
            ok = (np.abs(px / x_sd) <= x_cut)[None, :] & (np.abs(py / y_sd) <= y_cut)
            with np.errstate(all='ignore'):
                e = np.exp(-0.5 * ((px * px * xv)[None, :] + (px[None, :] * py * cov2) + (py * py * yv)))
            e = np.where(ok, e, 0.0)
            s = np.cumsum(e, axis=1)[:, -1] if e.shape[1] else np.zeros(y_n)
            dens[i] = (f * s).astype(np.float32)
        return dens

    def get_density_difference(self, members, to_complement, qm):
        if qm.TARGET_TYPE != TargetType.MULTI_NUMERIC:
            raise ValueError(str(qm) + " != " + str(TargetType.MULTI_NUMERIC))
        if members is None:
            raise ValueError("theSubgroup can not be null")
        if not members.any():
            raise ValueError("theSubgroup can not be empty")
        if qm == QM.L2:
            return float(F(self.l2(members, to_complement)))
        if len(self._grid) != 2:
            raise ValueError(str(qm) + " only implemented for 2D")
        return float(self.get_density_difference_2d(members, to_complement, qm)[0][0][0])

    def get_density_difference_2d(self, members, to_complement, qm):
        """grid set up from the subgroup and complement statistics; with qm
        only the summed divergence is returned in [0][0][0]"""
        data = self._data
        N = data.shape[0]
        m = np.asarray(members[:N], dtype=bool)
        st = _stats(data, m, True)
        s_size = int(st[0][SIZE_N])
        c_size = int(st[1][SIZE_N])
        s_cm = _bandwidth_matrix(st[0])
        c_cm = _bandwidth_matrix(st[1])
        if _is_degenerate(s_cm) or _is_degenerate(c_cm):
            print("\nERROR: degenerate covariance matrix: \n SG %s\n    %s\n!SG %s\n    %s"
                  % (_fstr(s_cm[0]), _fstr(s_cm[1]), _fstr(c_cm[0]), _fstr(c_cm[1])))
            return [[[0.0], None, None]]
        s_inv = _inverse(s_cm)
        c_inv = _inverse(c_cm)
        s_xv, s_cov2, s_yv = float(s_inv[0][0]), float(s_inv[0][1]) * 2.0, float(s_inv[1][1])
        c_xv, c_cov2, c_yv = float(c_inv[0][0]), float(c_inv[0][1]) * 2.0, float(c_inv[1][1])

        s_xs = jsqrt(float(s_cm[0][0]))
        s_ys = jsqrt(float(s_cm[1][1]))
        c_xs = jsqrt(float(c_cm[0][0]))
        c_ys = jsqrt(float(c_cm[1][1]))
        x_ext = CUTOFF * _jmax(s_xs, c_xs)
        x_min = self._limits[0][MIN] - x_ext
        x_max = self._limits[0][MAX] + x_ext
        y_ext = CUTOFF * _jmax(s_ys, c_ys)
        y_min = self._limits[1][MIN] - y_ext
        y_max = self._limits[1][MAX] + y_ext
        from .java_compat import jmin, jmax
        dx = jmin(s_xs, c_xs) / X_RESOLUTION
        dy = jmin(s_ys, c_ys) / Y_RESOLUTION
        dxdy = dx * dy
        x_range = x_max - x_min
        x_n = _floor_int(div(x_range, dx))
        y_range = y_max - y_min
        y_n = _floor_int(div(y_range, dy))
        self.last_dxdy = [x_min, x_max, x_n, y_min, y_max, y_n, dx, dy]

        _debug("\nSG")
        _debug("stats " + _dstr(st[0]))
        _debug("cm    " + _fstr(s_cm[0]) + "\n      " + _fstr(s_cm[1]))
        _debug("cm^-1 " + _fstr(s_inv[0]) + "\n      " + _fstr(s_inv[1]))
        _debug("!SG")
        _debug("stats " + _dstr(st[1]))
        _debug("cm    " + _fstr(c_cm[0]) + "\n      " + _fstr(c_cm[1]))
        _debug("cm^-1 " + _fstr(c_inv[0]) + "\n      " + _fstr(c_inv[1]))
        _debug("GRID")
        _debug("x_min=%f x_max=%f x_range=%f dx=%f x_n=%d (X_RESOLUTION=%f)" % (x_min, x_max, x_range, dx, x_n, X_RESOLUTION))
        _debug("y_min=%f y_max=%f y_range=%f dy=%f y_n=%d (Y_RESOLUTION=%f)" % (y_min, y_max, y_range, dy, y_n, Y_RESOLUTION))
        if (s_xs * s_ys < dxdy) or (c_xs * c_ys < dxdy):
            _debug("ERROR")
        _debug("NOTE if ((#_xsigma * #_ysigma) < dxdy) then #_intregral < 1.0")
        _debug("        dx=%f\t        dy=%f\t         dxdy=%f" % (dx, dy, dxdy))
        _debug(" SG_xsigma=%f\t SG_ysigma=%f\txsigma*ysigma=%f" % (s_xs, s_ys, s_xs * s_ys))
        _debug("!SG_xsigma=%f\t!SG_ysigma=%f\txsigma*ysigma=%f" % (c_xs, c_ys, c_xs * c_ys))

        qm_only = qm is not None
        s_f = div(dxdy, (2.0 * math.pi * jsqrt(float(_det(s_cm))) * s_size))
        c_f = div(dxdy, (2.0 * math.pi * jsqrt(float(_det(c_cm))) * c_size))
        px_all = data[:, 0].astype(np.float64)
        py_all = data[:, 1].astype(np.float64)
        ys = y_min + (np.arange(y_n) * dy)
        s_int = 0.0
        c_int = 0.0
        difference = 0.0
        pdf_s = None if qm_only else np.zeros((x_n, y_n), dtype=np.float32)
        pdf_c = None if qm_only else np.zeros((x_n, y_n), dtype=np.float32)
        pdf_d = None if qm_only else np.zeros((x_n, y_n), dtype=np.float32)
        not_m = ~m
        for i in range(x_n):
            x = x_min + (i * dx)
            px = x - px_all                     # N
            py = ys[:, None] - py_all[None, :]  # y_n x N
            with np.errstate(all='ignore'):
                a = px * px
                es = np.exp(-0.5 * ((a * s_xv)[None, :] + (px[None, :] * py * s_cov2) + (py * py * s_yv)))
                ec = np.exp(-0.5 * ((a * c_xv)[None, :] + (px[None, :] * py * c_cov2) + (py * py * c_yv)))
            es = np.where(m[None, :], es, 0.0)
            ec = np.where(not_m[None, :], ec, 0.0)
            s_kde = (np.cumsum(es, axis=1)[:, -1] if N else np.zeros(y_n)) * s_f
            c_kde = (np.cumsum(ec, axis=1)[:, -1] if N else np.zeros(y_n)) * c_f
            if not qm_only:
                pdf_s[i] = s_kde.astype(np.float32)
                pdf_c[i] = c_kde.astype(np.float32)
                pdf_d[i] = (s_kde - c_kde).astype(np.float32)
                s_int = _last_cumsum(s_kde, s_int)
                c_int = _last_cumsum(c_kde, c_int)
                continue
            div_vals = _divergence(qm, s_kde, c_kde, s_size, N)
            running = np.cumsum(np.concatenate(([difference], div_vals)))[1:]
            inf = np.flatnonzero(np.isinf(running))
            if inf.size:
                # no need to continue, java returns at the first infinite sum
                j = int(inf[0])
                difference = float(running[j])
                _debug("[%d,%d]=(%20.16f,%20.16f) %20.16f %20.16f" % (i, j, x, ys[j], s_kde[j], c_kde[j]))
                _debug("QM = " + double_to_string(difference))
                return [[[float(F(difference))], None, None]]
            s_int = _last_cumsum(s_kde, s_int)
            c_int = _last_cumsum(c_kde, c_int)
            if running.size:
                difference = float(running[-1])
        print("S_kde_integral = " + double_to_string(s_int))
        print("C_kde_integral = " + double_to_string(c_int))
        if qm_only:
            _debug("QM = " + double_to_string(difference))
            return [[[float(F(difference))], None, None]]
        return [pdf_s, pdf_c, pdf_d]

    def l2(self, members, to_complement):
        data = self._data
        N, D = data.shape
        m = np.asarray(members[:N], dtype=bool)
        s_size = int(np.count_nonzero(m))
        c_size = N - (s_size if to_complement else 0)
        not_m = ~m if to_complement else np.ones(N, dtype=bool)
        s_mu = _get_mu_vector(data, m)
        c_mu = _get_mu_vector(data, not_m)
        s_cm = _get_covariance_matrix(data, m, s_mu)
        c_cm = _get_covariance_matrix(data, not_m, c_mu)
        cm = (s_cm + c_cm).astype(np.float32)
        cm_inv = _inverse(cm)
        from .java_compat import int32
        f = div(1.0, (math.pow(2.0 * math.pi, D / 2.0) * jsqrt(float(F(_det(cm) * F(int32(s_size * c_size)))))))
        r = np.asarray(cm_inv, dtype=np.float64)
        others = data[not_m]
        total = 0.0
        for i in np.flatnonzero(m):
            diff = (data[i][None, :] - others).astype(np.float32).astype(np.float64)   # float subtraction
            M2 = np.zeros(diff.shape[0])
            for k in range(D):
                p = diff[:, k]
                M2 = M2 + p * p * r[k][k]
                for mm in range(k + 1, D):
                    M2 = M2 + 2.0 * p * diff[:, mm] * r[k][mm]
            total = _last_cumsum(np.exp(-0.5 * M2), total)
        return f * total


def _jmax(a, b):
    from .java_compat import jmax
    return jmax(a, b)


def _floor_int(v):
    """(int) Math.floor(v): NaN -> 0, clamps to the int range"""
    from .java_compat import to_int
    return to_int(math.floor(v)) if math.isfinite(v) else to_int(v)


def _get_data(columns):
    n = columns[0].size()
    return np.column_stack([c.floats_view()[:n] for c in columns]).astype(np.float32)


def _get_grid(columns, cutoff, nr_samples):
    out = []
    size = columns[0].size()
    b = np.ones(size, dtype=bool)
    for c in columns:
        mn = float(c.get_min())
        mx = float(c.get_max())
        ssd = float(c.get_statistics(None, b, False, False).get_subgroup_sum_squared_deviations())
        var = ssd / c.size()
        sigma = math.sqrt(var)
        ext = cutoff * sigma
        mn -= ext
        mx += ext
        rng = mx - mn
        n = math.floor((rng / sigma) * nr_samples) if sigma != 0 else float('inf')
        print("min=%f max=%f range=%f sigma=%f n=%f" % (mn, mx, rng, sigma, n))
        if n > 2147483647:
            raise IndexError("TOO MANY GRID POINTS")
        out.append([mn, mx, float(n)])
    return out


def _get_mu_vector(data, members):
    """float sums in row order, then divided by n"""
    sel = data[np.asarray(members[:data.shape[0]], dtype=bool)]
    n = sel.shape[0]
    mus = np.zeros(data.shape[1], dtype=np.float32)
    if n:
        mus = np.cumsum(sel, axis=0, dtype=np.float32)[-1]
    with np.errstate(all='ignore'):
        return (mus / F(n)).astype(np.float32)


def h_silverman(nr_dimensions, data_size):
    d = nr_dimensions
    n = float(data_size)
    p = 1.0 / (d + 4.0)
    d_fac = math.pow(4.0 / (d + 2.0), p)
    n_fac = math.pow(n, -p) if n > 0 else float('inf')
    return float(F(d_fac * n_fac))


def _std_dev(data, members, mus):
    sel = data[np.asarray(members[:data.shape[0]], dtype=bool)]
    with np.errstate(all='ignore'):
        f = (sel - mus).astype(np.float32)
        sq = (f * f).astype(np.float32)
        ssds = np.cumsum(sq, axis=0, dtype=np.float32)[-1] if sel.shape[0] else np.zeros(data.shape[1], np.float32)
        n = sel.shape[0] - 1
        return np.array([F(math.sqrt(float(F(v / F(n)))) if float(F(v / F(n))) >= 0 else float('nan')) for v in ssds], dtype=np.float32)


def _get_covariance_matrix(data, members, mus):
    """diagonal (Silverman) bandwidth matrix"""
    d = data.shape[1]
    h = F(h_silverman(d, int(np.count_nonzero(members))))
    sds = _std_dev(data, members, mus)
    cm = np.zeros((d, d), dtype=np.float32)
    for i in range(d):
        h_ii = F(h * sds[i])
        cm[i][i] = F(h_ii * h_ii)
    return cm


def _stats(data, members, is_sample):
    """single pass Welford statistics for subgroup and complement (2D)"""
    sg = [0.0] * NR_STATS
    co = [0.0] * NR_STATS
    xs = data[:, 0].astype(np.float64).tolist()
    ys = data[:, 1].astype(np.float64).tolist()
    mm = members.tolist()
    for i in range(len(xs)):
        s = sg if mm[i] else co
        x = xs[i]
        y = ys[i]
        s[SIZE_N] += 1.0
        n = s[SIZE_N]
        dx = x - s[MEAN_D1]
        dy = y - s[MEAN_D2]
        s[MEAN_D1] += dx / n
        s[MEAN_D2] += dy / n
        s[VAR_D1] += dx * (x - s[MEAN_D1])
        s[VAR_D2] += dy * (y - s[MEAN_D2])
        s[COV_1_2] = ((s[COV_1_2] * (n - 1.0)) + (dx * (y - s[MEAN_D2]))) / n
    for s in (sg, co):
        n = s[SIZE_N]
        d = n - (1.0 if is_sample else 0.0)
        s[VAR_D1] = div(s[VAR_D1], d)
        s[VAR_D2] = div(s[VAR_D2], d)
        if is_sample:
            s[COV_1_2] *= div(n, d)
        s[H] = math.pow(n, -(1.0 / 6.0)) if n > 0 else float('inf')
    return sg, co


def _bandwidth_matrix(stats):
    h = stats[H]
    h2 = h * h
    return np.array([[F(stats[VAR_D1] * h2), F(stats[COV_1_2] * h2)],
                     [F(stats[COV_1_2] * h2), F(stats[VAR_D2] * h2)]], dtype=np.float32)


def _det(m):
    m = np.asarray(m, dtype=np.float32)
    if m.shape[0] == 2:
        with np.errstate(all='ignore'):
            return F(F(m[0][0] * m[1][1]) - F(m[0][1] * m[1][0]))
    return F(np.linalg.det(m.astype(np.float64)))


def _inverse(m):
    m = np.asarray(m, dtype=np.float32)
    if m.shape[0] != m.shape[1]:
        raise ValueError("Not a squared matrix")
    if m.shape[0] == 2:
        f = _det(m)
        with np.errstate(all='ignore'):
            return np.array([[F(m[1][1] / f), F(-m[0][1] / f)],
                             [F(-m[1][0] / f), F(m[0][0] / f)]], dtype=np.float32)
    return np.linalg.inv(m.astype(np.float64)).astype(np.float32)


def _is_degenerate(m):
    return m[0][0] == 0.0 or m[1][1] == 0.0 or abs(m[0][1]) == 1.0


def _divergence(qm, p, q, coverage, nr_records):
    """vectorised over grid points, see the java note on 0 and Infinity"""
    with np.errstate(all='ignore'):
        if qm == QM.SQUARED_HELLINGER_2D:
            d = np.sqrt(p) - np.sqrt(q)
            return 0.5 * (d * d)
        if qm == QM.SQUARED_HELLINGER_WEIGHTED_2D:
            return (_divergence(QM.SQUARED_HELLINGER_2D, p, q, coverage, nr_records) * coverage) / nr_records
        if qm == QM.SQUARED_HELLINGER_WEIGHTED_ADJUSTED_2D:
            return _divergence(QM.SQUARED_HELLINGER_WEIGHTED_2D, p, q, coverage, nr_records) / (4.0 / 27.0)
        if qm == QM.KULLBACK_LEIBLER_2D:
            return np.where(p == 0.0, np.inf, p * np.log(p / q))
        if qm == QM.KULLBACK_LEIBLER_WEIGHTED_2D:
            return (_divergence(QM.KULLBACK_LEIBLER_2D, p, q, coverage, nr_records) * coverage) / nr_records
        if qm == QM.CWRACC_2D:
            return (_divergence(QM.CWRACC_UNWEIGHTED_2D, p, q, coverage, nr_records) * coverage) / nr_records
        if qm == QM.CWRACC_UNWEIGHTED_2D:
            return np.abs(q - p)
    raise AssertionError(qm)
