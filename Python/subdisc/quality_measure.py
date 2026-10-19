"""All quality measures.

The java class mixes float and double on purpose (well, historically), and
results are cast to float in many places. I copy every cast here, so the
qualities match the java version bit for bit where possible:
F(...) is a float32 value, plain python floats are doubles.
"""

import math

import numpy as np

from . import log
from . import function
from .java_compat import div, jsqrt, jlog, jexp, jpow, to_f32, fdiv, double_to_string, float_to_string
from .qm import QM
from .target_type import TargetType

F = np.float32
TEMPORARY_CODE_SG_VS_COMPLEMENT = False
FLOAT_MAX = float(np.finfo(np.float32).max)


def _f(x):
    return float(np.float32(x))


class QualityMeasure:
    # XXX MM (java) why are these static fields: they are shared by all
    # instances in java, and the port keeps it that way
    # MULTI_LABEL (Bayesian)
    _nr_nodes = 0
    _alpha = 0.0
    _beta = 0.0
    _v_structures = None
    # SCAPE
    _binary_target = None
    _numeric_target = None
    _descending_ordering_permutation = None
    _overall_subranking_loss = 0.0   # MUST BE 0.0f at first call

    def __init__(self, measure, total_coverage, *args):
        """Constructor overloads, picked on the arguments:

        QualityMeasure(qm, N, H)                         SINGLE_NOMINAL (+ double targets)
        QualityMeasure(qm, N, sum, ssd, pdf)             SINGLE_NUMERIC
        QualityMeasure(qm, N, label_ranking, lrm)        LABEL_RANKING
        QualityMeasure(qm, N, H, binary_col, num_col)    SCAPE
        QualityMeasure.multi_label(search_params, dag, N)
        """
        self._qm = measure
        self._nr_records = total_coverage
        self._total_target_coverage = 0
        self._total_average = float('nan')
        self._total_standard_deviation = float('nan')
        self._pdf = None
        self._dag = None
        self._average_ranking = None
        self._average_ranking_matrix = None

        from .label_ranking import LabelRanking
        if measure is None:
            raise ValueError("QualityMeasure: theMeasure can not be null")

        if len(args) == 1:
            self._init_nominal(measure, total_coverage, args[0])
        elif len(args) == 2 and isinstance(args[0], LabelRanking):
            self._init_label_ranking(measure, total_coverage, args[0], args[1])
        elif len(args) == 3 and hasattr(args[1], "get_type"):
            self._init_scape(measure, total_coverage, args[0], args[1], args[2])
        elif len(args) == 3:
            self._init_numeric(measure, total_coverage, args[0], args[1], args[2])
        elif len(args) == 0:
            pass
        else:
            raise TypeError("unknown QualityMeasure constructor")

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    def _init_nominal(self, measure, total_coverage, total_target_coverage):
        ok = (measure in QM.get_quality_measures(TargetType.SINGLE_NOMINAL)
              or measure in QM.get_quality_measures(TargetType.DOUBLE_REGRESSION)
              or measure in QM.get_quality_measures(TargetType.DOUBLE_CORRELATION)
              or measure in QM.get_quality_measures(TargetType.DOUBLE_BINARY))
        if not ok:
            raise ValueError("QualityMeasure: not a SINGLE_NOMINAL, DOUBLE_CORRELATION, DOUBLE_REGRESSION or DOUBLE_BINARY measure")
        if total_coverage <= 0:
            raise ValueError("QualityMeasure: theTotalCoverage must be > 0")
        if total_target_coverage <= 0:
            raise ValueError("QualityMeasure: theTotalTargetCoverage must be > 0")
        if total_coverage < total_target_coverage:
            raise ValueError("QualityMeasure: theTotalCoverage < theTotalTargetCoverage")
        self._total_target_coverage = total_target_coverage

    def _init_numeric(self, measure, total_coverage, total_sum, total_ssd, data_pdf):
        if measure not in QM.get_quality_measures(TargetType.SINGLE_NUMERIC):
            raise ValueError("QualityMeasure: not a SINGLE_NUMERIC measure")
        if total_coverage <= 0:
            raise ValueError("QualityMeasure: theTotalCoverage must be > 0")
        if math.isnan(total_sum) or math.isinf(total_sum):
            raise ValueError("QualityMeasure: theTotalSum can not be NaN or (-)Infinity")
        if math.isnan(total_ssd) or math.isinf(total_ssd) or total_ssd < 0.0:
            raise ValueError("QualityMeasure: theTotalSSD can not be NaN, (-)Infinity, or < 0.0f")
        if data_pdf is None:
            raise ValueError("QualityMeasure: theDataPDF can not be null")
        self._total_average = fdiv(total_sum, total_coverage)
        # float / int in java, then sqrt as double, back to float
        self._total_standard_deviation = to_f32(jsqrt(fdiv(total_ssd, total_coverage)))
        self._pdf = data_pdf

    def _init_label_ranking(self, measure, total_coverage, average_ranking, average_ranking_matrix):
        if measure not in QM.get_quality_measures(TargetType.LABEL_RANKING):
            raise ValueError("QualityMeasure: not a LabelRanking measure")
        if total_coverage <= 0:
            raise ValueError("QualityMeasure: theTotalCoverage must be > 0")
        if average_ranking is None:
            raise ValueError("QualityMeasure: theAverageRanking can not be null")
        if average_ranking_matrix is None:
            raise ValueError("QualityMeasure: theAverageRankingMatrix can not be null")
        self._average_ranking = average_ranking
        self._average_ranking_matrix = average_ranking_matrix
        log.log_command_line("average ranking of entire dataset:")
        average_ranking_matrix.print()

    @staticmethod
    def multi_label(search_parameters, dag, total_coverage):
        if search_parameters is None:
            raise ValueError("QualityMeasure: theSearchParameters can not be null")
        qm = search_parameters.get_quality_measure()
        if qm is None:
            raise ValueError("QualityMeasure: theSearchParameters.getQualityMeasure() can not be null")
        if qm not in QM.get_quality_measures(TargetType.MULTI_LABEL):
            raise ValueError("QualityMeasure: not a MULTI_LABEL measure")
        if dag is None:
            raise ValueError("QualityMeasure: theDAG can not be null")
        if total_coverage <= 0:
            raise ValueError("QualityMeasure: theTotalCoverage must be > 0")
        q = QualityMeasure(qm, total_coverage)
        q._dag = dag
        QualityMeasure._nr_nodes = dag.get_size()
        QualityMeasure._alpha = search_parameters.get_alpha()
        QualityMeasure._beta = search_parameters.get_beta()
        QualityMeasure._v_structures = dag.determine_v_structures()
        return q

    def _init_scape(self, measure, total_coverage, total_target_coverage, binary_target, numeric_target):
        if total_coverage <= 0:
            raise ValueError("QualityMeasure: theTotalCoverage must be > 0")
        if total_target_coverage <= 0:
            raise ValueError("QualityMeasure: theTotalTargetCoverage must be > 0")
        if binary_target is None:
            raise ValueError("QualityMeasure: theBinaryTarget can not be null")
        if numeric_target is None:
            raise ValueError("QualityMeasure: theNumericTarget can not be null")
        self._total_target_coverage = total_target_coverage
        QualityMeasure._binary_target = binary_target
        QualityMeasure._numeric_target = numeric_target
        QualityMeasure._descending_ordering_permutation = self._generate_ordering_permutation()
        self._set_average_subranking_loss()

    def get_probability_density_function(self):
        return self._pdf

    # ------------------------------------------------------------------
    # SINGLE_NOMINAL
    # ------------------------------------------------------------------

    def calculate(self, *args):
        """calculate(HB, B) for nominal targets, calculate(statistics, pdf)
        for numeric targets, calculate(subgroup) for multi-label and
        calculate(members, coverage, target_coverage) for SCAPE"""
        if len(args) == 2 and isinstance(args[0], (int, np.integer)):
            return self._calculate_nominal(int(args[0]), int(args[1]))
        if len(args) == 2:
            return self._calculate_numeric(args[0], args[1])
        if len(args) == 1:
            return self._calculate_multi_label(args[0])
        if len(args) == 3:
            return self._calculate_scape(args[0], args[1], args[2])
        raise TypeError("calculate()")

    def _calculate_nominal(self, count_head_body, coverage):
        if coverage <= 0:
            raise ValueError("QualityMeasure: theCoverage <= 0")
        if coverage > self._nr_records:
            raise ValueError("QualityMeasure: theCoverage > itsNrRecords")
        if count_head_body < 0:
            raise ValueError("QualityMeasure: theCountHeadBody < 0")
        if count_head_body < (coverage - (self._nr_records - self._total_target_coverage)):
            print("itsNrRecords " + str(self._nr_records))
            print("itsTotalTargetCoverage " + str(self._total_target_coverage))
            print("theCoverage " + str(coverage))
            print("theCountHeadBody " + str(count_head_body))
            raise ValueError("QualityMeasure: theCountHeadBody < theCoverage - (itsNrRecords - itsTotalTargetCoverage)")
        if count_head_body > self._total_target_coverage:
            raise ValueError("QualityMeasure: theCountHeadBody > itsTotalTargetCoverage")
        if count_head_body > coverage:
            raise ValueError("QualityMeasure: theCountHeadBody > theCoverage")
        return calculate_nominal(self._qm, self._nr_records, self._total_target_coverage, count_head_body, coverage)

    def get_nr_records(self):
        return self._nr_records

    def get_nr_positives(self):
        return self._total_target_coverage

    def get_roc_heaven(self):
        """quality of the upper left corner"""
        return self._calculate_nominal(self._total_target_coverage, self._total_target_coverage)

    def get_roc_hell(self):
        """lower right corner, calls the static calculate directly (B may be 0)"""
        return calculate_nominal(self._qm, self._nr_records, self._total_target_coverage, 0,
                                 self._nr_records - self._total_target_coverage)

    def get_base_label_ranking(self):
        return self._average_ranking

    def get_base_label_ranking_matrix(self):
        return self._average_ranking_matrix

    def compute_label_ranking_distance(self, support, subgroup_matrix):
        m = self._average_ranking_matrix
        q = self._qm
        if q == QM.LR_NORM:
            d = m.norm_distance(subgroup_matrix)
        elif q == QM.LR_NORM_MODE:
            d = m.norm_distance_mode(subgroup_matrix)
        elif q == QM.LR_WNORM:
            d = m.wnorm_distance(subgroup_matrix)
        elif q == QM.LR_LABELWISE_MIN:
            d = m.min_distance(subgroup_matrix)
        elif q == QM.LR_LABELWISE_MAX:
            d = m.labelwise_max(subgroup_matrix)
        elif q == QM.LR_PAIRWISE_MAX:
            d = m.pairwise_max(subgroup_matrix)
        elif q == QM.LR_COVARIANCE:
            d = m.cov_distance(subgroup_matrix)
        else:
            if q in QM.get_quality_measures(TargetType.LABEL_RANKING):
                raise AssertionError(q)
            raise ValueError("Invalid argument: " + str(q))
        percentage = fdiv(support, self._nr_records)
        size = to_f32(math.sqrt(percentage))
        return _f(F(size) * F(d))

    # ------------------------------------------------------------------
    # SINGLE_NUMERIC
    # ------------------------------------------------------------------

    def _calculate_numeric(self, stats, pdf):
        """FIXME (java) historic version with float casts, see calculate2()"""
        from . import subgroup_discovery as sd
        cov = stats.get_coverage()
        s = stats.get_subgroup_sum()
        ssd = stats.get_subgroup_sum_squared_deviations()
        med = stats.get_median()
        mad = stats.get_median_absolute_deviations()
        N = self._nr_records
        tavg = self._total_average
        tsd = self._total_standard_deviation
        q = self._qm
        with np.errstate(all='ignore'):
            avg = F(s) / F(cov)                     # float / int
            dev = float(F(avg - F(tavg)))           # float - float
            sqrt_cov = math.sqrt(cov) if cov >= 0 else float('nan')

            if q == QM.EXPLAINED_VARIANCE:
                if N <= 1:
                    return 0.0
                num = float(F(F(ssd) + F(stats.get_complement_sum_squared_deviations())))
                return _f(1.0 - div(num, (float(tsd) * float(tsd)) * N))
            if q == QM.Z_SCORE:
                return 0.0 if N <= 1 else _f(div(sqrt_cov * dev, tsd))
            if q == QM.INVERSE_Z_SCORE:
                return 0.0 if N <= 1 else _f(-div(sqrt_cov * dev, tsd))
            if q == QM.ABS_Z_SCORE:
                return 0.0 if N <= 1 else _f(abs(div(sqrt_cov * dev, tsd)))
            if q == QM.AVERAGE:
                return float(avg)
            if q == QM.INVERSE_AVERAGE:
                return float(F(-F(s)) / F(cov))
            if q == QM.QM_SUM:
                return float(F(s))
            if q == QM.INVERSE_SUM:
                return float(-F(s))
            if q == QM.ABS_DEVIATION:
                return abs(dev)
            if q == QM.MEAN_TEST:
                return _f(sqrt_cov * dev)
            if q == QM.INVERSE_MEAN_TEST:
                return _f(-(sqrt_cov * dev))
            if q == QM.ABS_MEAN_TEST:
                return _f(abs(sqrt_cov * dev))
            if q in (QM.T_TEST, QM.INVERSE_T_TEST, QM.ABS_T_TEST):
                if cov <= 1:
                    return 0.0
                t = div(sqrt_cov * dev, jsqrt(div(float(ssd), cov - 1.0)))
                if q == QM.T_TEST:
                    return _f(t)
                if q == QM.INVERSE_T_TEST:
                    return _f(-t)
                return _f(abs(t))
            # ORDINAL
            if q == QM.AUC:
                compl = F(N - cov)
                seq = F(F(F(cov) * F(cov + 1.0)) / F(2.0))
                return float(F(1.0) + F(F(seq - F(s)) / F(F(cov) * compl)))
            if q in (QM.WMW_RANKS, QM.INVERSE_WMW_RANKS, QM.ABS_WMW_RANKS):
                compl = F(N - cov)
                mean = F(F(F(cov) * F(F(F(cov) + compl) + F(1))) / F(2.0))
                sd_ = F(math.sqrt(float(F(F(F(F(cov) * compl) * F(F(F(cov) + compl) + F(1))) / F(12.0)))))
                r = F(F(F(s) - mean) / sd_)
                if q == QM.WMW_RANKS:
                    return float(r)
                if q == QM.INVERSE_WMW_RANKS:
                    return float(-r)
                return abs(float(r))
            if q == QM.MMAD:
                return float(F(F(cov) / F(F(F(2.0) * F(med)) + F(mad))))
            # DISTRIBUTION
            if q in (QM.SQUARED_HELLINGER, QM.SQUARED_HELLINGER_WEIGHTED, QM.SQUARED_HELLINGER_WEIGHTED_ADJUSTED):
                if q == QM.SQUARED_HELLINGER and sd.TEMPORARY_CODE:
                    return _f(pdf.score)
                total = 0.0
                base = self._pdf
                for i in range(base.size()):
                    if q == QM.SQUARED_HELLINGER and TEMPORARY_CODE_SG_VS_COMPLEMENT:
                        dens = pdf.get_complement_density(i)
                    else:
                        dens = base.get_density(i)
                    d = jsqrt(dens) - jsqrt(pdf.get_density(i))
                    total += d * d
                log.log_command_line("difference in PDF: " + double_to_string(total))
                if q == QM.SQUARED_HELLINGER:
                    return _f(0.5 * total)
                r = _f(div(0.5 * (total * cov), N))
                if q == QM.SQUARED_HELLINGER_WEIGHTED:
                    return r
                # magic number = maximum possible score, at (5/9, 4/27)
                return _f(r / (4.0 / 27.0))
            if q in (QM.KULLBACK_LEIBLER, QM.KULLBACK_LEIBLER_WEIGHTED):
                total = 0.0
                base = self._pdf
                for i in range(base.size()):
                    dens = base.get_density(i)
                    sub = pdf.get_density(i)
                    # 0 by definition, also avoids a divide by 0
                    if sub == 0.0:
                        continue
                    total += sub * jlog(fdiv(sub, dens))
                if q == QM.KULLBACK_LEIBLER:
                    return _f(total)
                return _f(div(total * cov, N))
            if q in (QM.CWRACC, QM.CWRACC_UNWEIGHTED):
                total = 0.0
                base = self._pdf
                for i in range(base.size()):
                    total += abs(float(F(F(base.get_density(i)) - F(pdf.get_density(i)))))
                log.log_command_line("difference in PDF: " + double_to_string(total))
                if q == QM.CWRACC:
                    return _f(div(total * cov, N))
                return _f(total)
        if q in QM.get_quality_measures(TargetType.SINGLE_NUMERIC) or q in QM.get_quality_measures(TargetType.SINGLE_ORDINAL):
            raise AssertionError(q)
        raise ValueError("Invalid QM: " + str(q))

    def calculate2(self, stats, pdf):
        """FOR FUTURE USE (java): the same measures in plain double arithmetic"""
        cov = float(stats.get_coverage())
        s = float(stats.get_subgroup_sum())
        ssd = float(stats.get_subgroup_sum_squared_deviations())
        med = float(stats.get_median())
        mad = float(stats.get_median_absolute_deviations())
        N = self._nr_records
        tavg = float(self._total_average)
        tsd = float(self._total_standard_deviation)
        q = self._qm
        dev = div(s, cov) - tavg
        if q == QM.EXPLAINED_VARIANCE:
            return 0.0 if N <= 1 else 1.0 - div(ssd + stats.get_complement_sum_squared_deviations(), tsd * tsd * N)
        if q == QM.Z_SCORE:
            return 0.0 if N <= 1 else div(jsqrt(cov) * dev, tsd)
        if q == QM.INVERSE_Z_SCORE:
            return 0.0 if N <= 1 else -div(jsqrt(cov) * dev, tsd)
        if q == QM.ABS_Z_SCORE:
            return 0.0 if N <= 1 else abs(div(jsqrt(cov) * dev, tsd))
        if q == QM.AVERAGE:
            return div(s, cov)
        if q == QM.INVERSE_AVERAGE:
            return div(-s, cov)
        if q == QM.QM_SUM:
            return s
        if q == QM.INVERSE_SUM:
            return -s
        if q == QM.ABS_DEVIATION:
            return abs(dev)
        if q == QM.MEAN_TEST:
            return jsqrt(cov) * dev
        if q == QM.INVERSE_MEAN_TEST:
            return -(jsqrt(cov) * dev)
        if q == QM.ABS_MEAN_TEST:
            return abs(jsqrt(cov) * dev)
        if q in (QM.T_TEST, QM.INVERSE_T_TEST, QM.ABS_T_TEST):
            if cov <= 1:
                return 0.0
            t = div(jsqrt(cov) * dev, jsqrt(div(ssd, cov - 1.0)))
            return t if q == QM.T_TEST else (-t if q == QM.INVERSE_T_TEST else abs(t))
        if q == QM.AUC:
            compl = N - cov
            seq = cov * (cov + 1.0) / 2.0
            return 1.0 + div(seq - s, cov * compl)
        if q in (QM.WMW_RANKS, QM.INVERSE_WMW_RANKS, QM.ABS_WMW_RANKS):
            compl = N - cov
            mean = (cov * (cov + compl + 1.0)) / 2.0
            sd_ = jsqrt((cov * compl * (cov + compl + 1.0)) / 12.0)
            r = div(s - mean, sd_)
            return r if q == QM.WMW_RANKS else (-r if q == QM.INVERSE_WMW_RANKS else abs(r))
        if q == QM.MMAD:
            return div(cov, 2.0 * med + mad)
        return self._calculate_numeric(stats, pdf)

    # ------------------------------------------------------------------
    # MULTI_LABEL
    # ------------------------------------------------------------------

    def _calculate_multi_label(self, subgroup):
        q = self._qm
        if q == QM.WEED:
            return _f(jpow(calculate_entropy(self._nr_records, subgroup.get_coverage()), QualityMeasure._alpha)
                      * jpow(self.calculate_edit_distance(subgroup.get_dag()), QualityMeasure._beta))
        if q == QM.EDIT_DISTANCE:
            return self.calculate_edit_distance(subgroup.get_dag())
        if q in QM.get_quality_measures(TargetType.MULTI_LABEL):
            raise AssertionError(q)
        raise ValueError("Invalid QM: " + str(q))

    def calculate_edit_distance(self, dag):
        n = QualityMeasure._nr_nodes
        if dag.get_size() != n:
            log.log_command_line("Comparing incompatible DAG's. One has " + str(dag.get_size()) + " nodes and the other has "
                                 + str(n) + ". Throw pi.")
            return _f(math.pi)
        edits = 0
        vs = QualityMeasure._v_structures
        for i in range(n):
            for j in range(i):
                a = dag.get_node(j).is_connected(i)
                b = self._dag.get_node(j).is_connected(i)
                if (a == 0 and b != 0) or (a != 0 and b == 0):
                    edits += 1
                elif a == 0:
                    if dag.test_v_structure(j, i) != vs[j][i]:
                        edits += 1
        # n choose 2, int division in java
        return fdiv(edits, (n * (n - 1)) // 2)

    # ------------------------------------------------------------------
    # SCAPE
    # ------------------------------------------------------------------

    def _generate_ordering_permutation(self):
        """indices sorting the numeric target descending, ties by index"""
        from .column import _order_keys
        vals = QualityMeasure._numeric_target.get_floats()[:self._nr_records]
        keys = _order_keys(vals)
        return np.argsort(-keys, kind='stable')

    def _set_average_subranking_loss(self):
        whole = np.ones(self._nr_records, dtype=bool)
        QualityMeasure._overall_subranking_loss = self._calculate_scape(whole, self._nr_records, self._total_target_coverage)

    def get_overall_subranking_loss(self):
        return QualityMeasure._overall_subranking_loss

    def _calculate_scape(self, members, coverage, target_coverage):
        if self._nr_records <= 1:
            return 0.0
        q = self._qm
        if q == QM.SUBRANKING_LOSS:
            return self.calculate_subranking_loss(members, coverage, target_coverage)
        if q == QM.NEGATIVE_SUBRANKING_LOSS:
            return -self.calculate_subranking_loss(members, coverage, target_coverage)
        if q == QM.RELATIVE_SUBRANKING_LOSS:
            return _f(F(self.calculate_subranking_loss(members, coverage, target_coverage)) - F(QualityMeasure._overall_subranking_loss))
        if q == QM.REVERSE_RELATIVE_SUBRANKING_LOSS:
            return _f(F(QualityMeasure._overall_subranking_loss) - F(self.calculate_subranking_loss(members, coverage, target_coverage)))
        if q in QM.get_quality_measures(TargetType.SCAPE):
            raise AssertionError(q)
        raise ValueError("Invalid QM: " + str(q))

    def calculate_subranking_loss(self, members, coverage, nr_positives):
        """ranking loss with ties punished by 1/2, in float32 like java"""
        nr_negatives = coverage - nr_positives
        pos_loop = 0
        neg_loop = 0
        total = F(0.0)
        pos_ties = 0
        neg_ties = 0
        prev = FLOAT_MAX
        perm = QualityMeasure._descending_ordering_permutation
        nf = QualityMeasure._numeric_target.floats_view()
        bins = QualityMeasure._binary_target.binaries_view()
        cont = True
        i = 0
        n = self._nr_records
        with np.errstate(all='ignore'):
            while cont and i < n:
                idx = int(perm[i])
                cur = float(nf[idx])
                if cur < prev:
                    total = F(total + F(F(F(pos_ties * neg_ties) * F(0.5))))
                    prev = cur
                    pos_loop += pos_ties
                    neg_loop += neg_ties
                    pos_ties = 0
                    neg_ties = 0
                    if pos_loop == nr_positives:
                        cont = False
                    if neg_loop == nr_negatives:
                        cont = False
                        total = F(total + F((nr_positives - pos_loop) * nr_negatives))
                if members[idx] and cont:
                    if bins[idx]:
                        pos_ties += 1
                        total = F(total + F(neg_loop))
                    else:
                        neg_ties += 1
                i += 1
            return float(F(total / F(nr_positives)))


# ----------------------------------------------------------------------
# static helpers
# ----------------------------------------------------------------------

def calculate_nominal(measure, total_coverage, total_target_coverage, count_head_body, coverage):
    """the contingency table measures, all in double"""
    # FIXME MM (java) if (H==N) -> return 0.0
    if total_target_coverage == total_coverage or total_target_coverage == 0:
        return 0.0   # perhaps not ideal for some QMs, better than an exception
    N = float(total_coverage)
    H = float(total_target_coverage)
    B = float(coverage)
    HB = float(count_head_body)
    nHB = B - HB
    HnB = H - HB
    nHnB = (N - H) - nHB

    q = measure
    r = float('nan')
    if q == QM.WRACC:
        r = (HB / N) - (H / N) * (B / N)
    elif q == QM.ABSWRACC:
        r = abs((HB / N) - (H / N) * (B / N))
    elif q == QM.MUTUAL_INFORMATION:
        p_hb = HB / N
        p_nhb = nHB / N
        p_hnb = HnB / N
        p_nhnb = nHnB / N
        r = (_mi(p_hb, p_hnb, p_nhb) + _mi(p_nhb, p_nhnb, p_hb)
             + _mi(p_hnb, p_hb, p_nhnb) + _mi(p_nhnb, p_nhb, p_hnb))
    elif q == QM.CORTANA_QUALITY:
        quot = H / N
        wracc = (HB / N) - (quot * (B / N))
        max_wracc = quot - (quot * quot)
        r = div(wracc, max_wracc)
    elif q == QM.CHI_SQUARED:
        r = _chi_squared(N, H, B, HB)
    elif q == QM.INFORMATION_GAIN:
        r = _information_gain(N, H, B, HB)
    elif q == QM.BINOMIAL:
        r = 0.0 if B == 0 else math.sqrt(B / N) * (HB / B - H / N)
    elif q == QM.JACCARD:
        r = div(HB, B + HnB)
    elif q == QM.COVERAGE:
        r = B
    elif q == QM.TARGET_SHARE:
        r = 0.0 if B == 0 else HB / B
    elif q == QM.SPECIFICITY:
        r = div(nHnB, N - H)
        if math.isnan(r):
            r = 0.0
    elif q == QM.SENSITIVITY:
        r = HB / H
    elif q == QM.LAPLACE:
        r = (HB + 1) / (B + 2)
    elif q == QM.F_MEASURE:
        r = div(HB, B + H)
    elif q == QM.G_MEASURE:
        r = div(HB, nHB + H)
    elif q == QM.CORRELATION:
        nH = N - H
        r = div(HB * nH - H * nHB, jsqrt(H * nH * B * (N - B)))
        if math.isnan(r):
            r = 0.0
    elif q == QM.PURITY:
        r = 0.0 if B == 0 else HB / B
        if r < 0.5:
            r = 1.0 - r
    elif q == QM.BAYESIAN_SCORE:
        r = _bayesian_factor(N, H, B, HB)
    elif q == QM.LIFT:
        r = 0.0 if B == 0 else div(HB * N, B * H)
    elif q == QM.RELATIVE_LIFT:
        r = 0.0 if B == 0 else div(HB * N, B * H) - 1
    else:
        if q in QM.get_quality_measures(TargetType.SINGLE_NOMINAL):
            raise AssertionError(q)
        raise ValueError("Invalid argument: " + str(q))
    if math.isnan(r):
        raise AssertionError("QualityMeasure.calculate() NaN return")
    return r


def _mi(a, b, c):
    # by definition 0*log(x) = 0
    if a == 0.0:
        return 0.0
    return a * jlog(div(a, (a + b) * (a + c)))


def calculate_propensity_based(measure, count_head_body, coverage, total_count, count_head_propensity_score):
    hb = F(count_head_body)
    cov = F(coverage)
    tot = F(total_count)
    ps = F(count_head_propensity_score)
    with np.errstate(all='ignore'):
        if measure == QM.PROP_SCORE_WRACC:
            r = F(F(F(F(hb / cov) - F(ps / cov)) * cov) / tot)
            print("Calculate Propensity based WRAcc")
            print(float_to_string(r))
            return float(r)
        if measure == QM.PROP_SCORE_RATIO:
            return float(F(F(hb / tot) / F(ps / tot)))
    raise ValueError("QM invalid: " + str(measure))


def _chi_squared(total, head, body, head_body):
    if body == total or body == 0.0:
        return 0.0
    e = _expectency(total, body, head)
    q = div(_sq(head_body - e), e)
    e = _expectency(total, total - body, head)
    q += div(_sq(head - head_body - e), e)
    e = _expectency(total, total - head, body)
    q += div(_sq(body - head_body - e), e)
    e = _expectency(total, total - body, total - head)
    q += div(_sq((total - head - body + head_body) - e), e)
    return q


def _sq(v):
    return v * v


def _expectency(total, body, head):
    return total * (body / total) * (head / total)


def _lg(p):
    return jlog(p) / math.log(2.0)


def calculate_entropy(body, head_body):
    body = float(body)
    head_body = float(head_body)
    if body == 0.0:
        return 0.0   # special case that should never occur
    if head_body == 0.0 or body == head_body:
        return 0.0   # by definition
    pj = head_body / body
    return -1.0 * pj * _lg(pj) - (1.0 - pj) * _lg(1.0 - pj)


def _conditional_entropy(body, head_body):
    if body == 0.0:
        return 0.0
    phb = head_body / body
    pnhb = (body - head_body) / body
    if phb == 0.0 or pnhb == 0.0:
        return 0.0
    return -1.0 * phb * _lg(phb) - pnhb * _lg(pnhb)


def _information_gain(total, head, body, head_body):
    frac = body / total
    not_body = total - body
    head_not_body = head - head_body
    return (calculate_entropy(total, head)
            - (frac * _conditional_entropy(body, head_body))
            - ((1.0 - frac) * _conditional_entropy(not_body, head_not_body)))


def _bayesian_factor(total, head, body, head_body):
    """Iyad Batal: Bayes factor with uniform beta priors"""
    n11 = int(head_body)
    n12 = int(body - head_body)
    n21 = int(head - head_body)
    n22 = int(total - n11 - n12 - n21)
    n1 = n11 + n21
    n2 = n12 + n22
    alpha = beta = 1
    alpha1 = beta1 = 1
    alpha2 = beta2 = 1
    log_m_e = _score_m_e(n1, n2, alpha, beta)
    log_m_h = _score_m_h(n11, n12, n21, n22, alpha1, beta1, alpha2, beta2)[0]
    log_m_l = _score_m_h(n21, n22, n11, n12, alpha2, beta2, alpha1, beta1)[0]
    prior_m_e, prior_m_h, prior_m_l = 0.33333333, 0.33333333, 0.33333334
    log_numerator = math.log(prior_m_h) + log_m_h
    log_denom1 = _log_add(math.log(prior_m_e) + log_m_e, math.log(prior_m_l) + log_m_l)
    return log_numerator - log_denom1


def _log_add(x, y):
    if abs(x - y) >= 36.043:
        return max(x, y)
    return jlog(1 + jexp(y - x)) + x


def _log_diff(x, y):
    if (x - y) >= 36.043:
        return x
    return jlog(1 - jexp(y - x)) + x


def _score_m_e(n1, n2, alpha, beta):
    lg = function.log_gamma_big
    return lg(alpha + beta) - lg(alpha + n1 + beta + n2) + lg(alpha + n1) - lg(alpha) + lg(beta + n2) - lg(beta)


def _score_m_h(n11, n12, n21, n22, alpha1, beta1, alpha2, beta2):
    lg = function.log_gamma_big
    a = n21 + alpha2
    b = n22 + beta2
    c = n11 + alpha1
    d = n12 + beta1
    k = 0.5
    C = (lg(alpha1 + beta1) - lg(alpha1) - lg(beta1) + lg(alpha2 + beta2) - lg(alpha2) - lg(beta2))
    part2 = 0.0
    for i in range(1, b + 1):
        j = a + i - 1
        temp = (lg(a) + lg(b) - lg(j + 1) - lg(a + b - j) + lg(c + j) + lg(a + b + d - 1 - j) - lg(a + b + c + d - 1))
        part2 = temp if i == 1 else _log_add(part2, temp)
    part1 = lg(a) + lg(b) - lg(a + b) + lg(c) + lg(d) - lg(c + d)
    r0 = -math.log(k) + C + part2
    r1 = _log_diff(-math.log(k) + C + part1, r0)
    return r0, r1
