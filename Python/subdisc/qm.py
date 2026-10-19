from enum import Enum

from .stat_enum import Stat
from .target_type import TargetType


class QM(Enum):
    # (ordinal, gui text, default measure minimum, target type)

    # SINGLE_NOMINAL quality measures
    # NOTE when adding a new SINGLE_NOMINAL QM add it to get_definition()
    CORTANA_QUALITY = (0, "Cortana Quality", "0.1", TargetType.SINGLE_NOMINAL)
    WRACC = (1, "WRAcc", "0.02", TargetType.SINGLE_NOMINAL)
    BINOMIAL = (2, "Binomial Test", "0.05", TargetType.SINGLE_NOMINAL)
    RELATIVE_LIFT = (3, "Relative Lift", "0.0", TargetType.SINGLE_NOMINAL)
    ABSWRACC = (4, "Abs WRAcc", "0.02", TargetType.SINGLE_NOMINAL)
    CHI_SQUARED = (5, "Chi-squared", "50", TargetType.SINGLE_NOMINAL)
    INFORMATION_GAIN = (6, "Information Gain", "0.02", TargetType.SINGLE_NOMINAL)
    MUTUAL_INFORMATION = (7, "Mutual Information", "0.01", TargetType.SINGLE_NOMINAL)
    TARGET_SHARE = (8, "Target Share", "0.0", TargetType.SINGLE_NOMINAL)
    PURITY = (9, "Purity", "0.5", TargetType.SINGLE_NOMINAL)
    JACCARD = (10, "Jaccard", "0.2", TargetType.SINGLE_NOMINAL)
    COVERAGE = (11, "Coverage", "10", TargetType.SINGLE_NOMINAL)
    SPECIFICITY = (12, "Specificity", "0.5", TargetType.SINGLE_NOMINAL)
    SENSITIVITY = (13, "Sensitivity", "0.5", TargetType.SINGLE_NOMINAL)
    LAPLACE = (14, "Laplace", "0.2", TargetType.SINGLE_NOMINAL)
    F_MEASURE = (15, "F-measure", "0.2", TargetType.SINGLE_NOMINAL)
    G_MEASURE = (16, "G-measure", "0.2", TargetType.SINGLE_NOMINAL)
    CORRELATION = (17, "Correlation", "0.1", TargetType.SINGLE_NOMINAL)
    PROP_SCORE_WRACC = (18, "Propensity Score WRacc", "-0.25", TargetType.SINGLE_NOMINAL)
    PROP_SCORE_RATIO = (19, "Propensity Score Ratio", "1.0", TargetType.SINGLE_NOMINAL)
    BAYESIAN_SCORE = (20, "Bayesian Score", "0.0", TargetType.SINGLE_NOMINAL)
    LIFT = (21, "Lift", "1.0", TargetType.SINGLE_NOMINAL)

    # SINGLE_NUMERIC quality measures
    # NOTE when adding a new SINGLE_NUMERIC QM add it to required_stats()
    Z_SCORE = (22, "Z-Score", "1.0", TargetType.SINGLE_NUMERIC)
    EXPLAINED_VARIANCE = (23, "Explained Variance", "0.0", TargetType.SINGLE_NUMERIC)
    INVERSE_Z_SCORE = (24, "Inverse Z-Score", "1.0", TargetType.SINGLE_NUMERIC)
    ABS_Z_SCORE = (25, "Abs Z-Score", "1.0", TargetType.SINGLE_NUMERIC)
    AVERAGE = (26, "Average", "0.0", TargetType.SINGLE_NUMERIC)   # bogus minimum, should come from data
    INVERSE_AVERAGE = (27, "Inverse Average", "0.0", TargetType.SINGLE_NUMERIC)
    QM_SUM = (28, "Sum", "0.0", TargetType.SINGLE_NUMERIC)
    INVERSE_SUM = (29, "Inverse Sum", "0.0", TargetType.SINGLE_NUMERIC)
    ABS_DEVIATION = (30, "Abs Deviation", "0.0", TargetType.SINGLE_NUMERIC)
    MEAN_TEST = (31, "Mean Test", "0.01", TargetType.SINGLE_NUMERIC)
    INVERSE_MEAN_TEST = (32, "Inverse Mean Test", "0.01", TargetType.SINGLE_NUMERIC)
    ABS_MEAN_TEST = (33, "Abs Mean Test", "0.01", TargetType.SINGLE_NUMERIC)
    T_TEST = (34, "t-Test", "1.0", TargetType.SINGLE_NUMERIC)
    INVERSE_T_TEST = (35, "Inverse t-Test", "1.0", TargetType.SINGLE_NUMERIC)
    ABS_T_TEST = (36, "Abs t-Test", "1.0", TargetType.SINGLE_NUMERIC)
    SQUARED_HELLINGER = (37, "Squared Hellinger distance", "0.0", TargetType.SINGLE_NUMERIC)
    SQUARED_HELLINGER_WEIGHTED = (38, "Weighted Squared Hellinger distance", "0.0", TargetType.SINGLE_NUMERIC)
    SQUARED_HELLINGER_WEIGHTED_ADJUSTED = (39, "Adjusted Squared Hellinger distance", "0.0", TargetType.SINGLE_NUMERIC)
    KULLBACK_LEIBLER = (40, "Kullback-Leibler divergence", "0.0", TargetType.SINGLE_NUMERIC)
    KULLBACK_LEIBLER_WEIGHTED = (41, "Weighted Kullback-Leibler divergence", "0.0", TargetType.SINGLE_NUMERIC)
    CWRACC = (42, "CWRAcc", "0.0", TargetType.SINGLE_NUMERIC)
    CWRACC_UNWEIGHTED = (43, "CWRAcc Unweighted", "0.0", TargetType.SINGLE_NUMERIC)

    # MULTI_NUMERIC quality measures
    SQUARED_HELLINGER_2D = (44, "Squared Hellinger distance 2D", "0.0", TargetType.MULTI_NUMERIC)
    SQUARED_HELLINGER_WEIGHTED_2D = (45, "Weighted Squared Hellinger distance 2D", "0.0", TargetType.MULTI_NUMERIC)
    SQUARED_HELLINGER_WEIGHTED_ADJUSTED_2D = (46, "Adjusted Squared Hellinger distance 2D", "0.0", TargetType.MULTI_NUMERIC)
    KULLBACK_LEIBLER_2D = (47, "Kullback-Leibler divergence 2D", "0.0", TargetType.MULTI_NUMERIC)
    KULLBACK_LEIBLER_WEIGHTED_2D = (48, "Weighted Kullback-Leibler divergence 2D", "0.0", TargetType.MULTI_NUMERIC)
    CWRACC_2D = (49, "CWRAcc_2D", "0.0", TargetType.MULTI_NUMERIC)
    CWRACC_UNWEIGHTED_2D = (50, "CWRAcc Unweighted_2D", "0.0", TargetType.MULTI_NUMERIC)
    L2 = (51, "L2", "0.0", TargetType.MULTI_NUMERIC)

    # SINGLE_ORDINAL quality measures
    AUC = (52, "AUC of ROC", "0.5", TargetType.SINGLE_ORDINAL)
    WMW_RANKS = (53, "WMW-Ranks test", "1.0", TargetType.SINGLE_ORDINAL)
    INVERSE_WMW_RANKS = (54, "Inverse WMW-Ranks test", "1.0", TargetType.SINGLE_ORDINAL)
    ABS_WMW_RANKS = (55, "Abs WMW-Ranks test", "1.0", TargetType.SINGLE_ORDINAL)
    MMAD = (56, "Median MAD metric", "0", TargetType.SINGLE_ORDINAL)

    # MULTI_LABEL quality measures
    WEED = (57, "Wtd Ent Edit Dist", "0", TargetType.MULTI_LABEL)
    EDIT_DISTANCE = (58, "Edit Distance", "0", TargetType.MULTI_LABEL)

    # LABEL_RANKING
    LR_NORM = (59, "LRM Norm", "0.0", TargetType.LABEL_RANKING)
    LR_NORM_MODE = (60, "LRM Norm Mode", "0.0", TargetType.LABEL_RANKING)
    LR_WNORM = (61, "LRM Norm & homog", "0.0", TargetType.LABEL_RANKING)
    LR_LABELWISE_MIN = (62, "Labelwise Minimum", "0.0", TargetType.LABEL_RANKING)
    LR_LABELWISE_MAX = (63, "Labelwise Maximization", "0.0", TargetType.LABEL_RANKING)
    LR_PAIRWISE_MAX = (64, "Pairwise Maximization", "0.0", TargetType.LABEL_RANKING)
    LR_COVARIANCE = (65, "Covariance", "0.0", TargetType.LABEL_RANKING)

    # DOUBLE_CORRELATION quality measures
    CORRELATION_R = (66, "r", "0.2", TargetType.DOUBLE_CORRELATION)
    CORRELATION_R_NEG = (67, "Negative r", "0.2", TargetType.DOUBLE_CORRELATION)
    CORRELATION_R_NEG_SQ = (68, "Neg Sqr r", "0.2", TargetType.DOUBLE_CORRELATION)
    CORRELATION_R_SQ = (69, "Squared r", "0.2", TargetType.DOUBLE_CORRELATION)
    CORRELATION_DISTANCE = (70, "Distance", "0.0", TargetType.DOUBLE_CORRELATION)
    CORRELATION_P = (71, "p-Value Distance", "0.0", TargetType.DOUBLE_CORRELATION)
    CORRELATION_ENTROPY = (72, "Wtd Ent Distance", "0.0", TargetType.DOUBLE_CORRELATION)
    ADAPTED_WRACC = (73, "Adapted WRAcc", "0.0", TargetType.DOUBLE_CORRELATION)
    COSTS_WRACC = (74, "Costs WRAcc", "0.0", TargetType.DOUBLE_CORRELATION)
    CWTPD = (75, "CWTPD", "0.0", TargetType.DOUBLE_CORRELATION)
    TMCC = (76, "TMCC", "0.0", TargetType.DOUBLE_CORRELATION)
    MCC = (77, "MCC", "0.0", TargetType.DOUBLE_CORRELATION)
    PDC = (78, "PDC", "0.0", TargetType.DOUBLE_CORRELATION)
    MVPDC = (79, "MVPDC", "0.0", TargetType.DOUBLE_CORRELATION)

    # DOUBLE_BINARY
    RELATIVE_WRACC = (80, "Relative WRAcc", "0.0", TargetType.DOUBLE_BINARY)
    ABSOLUTE_WRACC = (81, "Absolute WRAcc", "0.0", TargetType.DOUBLE_BINARY)
    RELATIVE_RISK = (82, "Relative Risk", "0.0", TargetType.DOUBLE_BINARY)
    ABSOLUTE_RISK = (83, "Absolute Risk", "0.0", TargetType.DOUBLE_BINARY)

    # DOUBLE_REGRESSION quality measures
    REGRESSION_SSD_COMPLEMENT = (84, "Sign. of Slope Diff. (complement)", "0.0", TargetType.DOUBLE_REGRESSION)
    REGRESSION_SSD_DATASET = (85, "Sign. of Slope Diff. (dataset)", "0.0", TargetType.DOUBLE_REGRESSION)
    REGRESSION_FLATNESS = (86, "Flatness", "0.0", TargetType.DOUBLE_REGRESSION)
    REGRESSION_SSD_4 = (87, "Sign. of Slope Diff. 4", "0.0", TargetType.DOUBLE_REGRESSION)
    COOKS_DISTANCE = (88, "Cook's Distance", "0.0", TargetType.DOUBLE_REGRESSION)

    # SCAPE quality measures
    SUBRANKING_LOSS = (89, "Subranking loss", "0.0", TargetType.SCAPE)
    # WD: I needed something fierce, so here's Ernie's favorite number, but negative
    NEGATIVE_SUBRANKING_LOSS = (90, "Negative subranking loss", "-8243721.5", TargetType.SCAPE)
    RELATIVE_SUBRANKING_LOSS = (91, "Relative subranking loss", "0.0", TargetType.SCAPE)
    # relative to the ranking loss in the overall dataset, so 0 will do as minimum
    REVERSE_RELATIVE_SUBRANKING_LOSS = (92, "Reverse relative subranking loss", "0.0", TargetType.SCAPE)

    @property
    def GUI_TEXT(self):
        return self.value[1]

    @property
    def MEASURE_DEFAULT(self):
        return self.value[2]

    @property
    def TARGET_TYPE(self):
        return self.value[3]

    @property
    def ordinal(self):
        return self.value[0]

    def __str__(self):
        return self.value[1]

    def __lt__(self, other):
        return self.value[0] < other.value[0]

    @staticmethod
    def get_quality_measures(target_type):
        result = [qm for qm in QM if qm.TARGET_TYPE == target_type]
        # remove non implemented ones
        for qm in (QM.PROP_SCORE_WRACC, QM.PROP_SCORE_RATIO, QM.COOKS_DISTANCE):
            if qm in result:
                result.remove(qm)
        return result

    @staticmethod
    def from_string(text):
        """QM for the gui text, None when nothing fits."""
        for qm in QM:
            if qm.GUI_TEXT.lower() == (text or "").lower():
                return qm
        return None

    @staticmethod
    def get_definition(qm):
        """GnuPlot definition of a SINGLE_NOMINAL QM, see ROCCurveWindow.get_gnu_plot_string()."""
        s = qm.GUI_TEXT.replace(" ", "_").replace("-", "_") + "(x,y) = "
        d = _DEFINITIONS.get(qm)
        if d is not None:
            return s + d
        if qm.TARGET_TYPE == TargetType.SINGLE_NOMINAL:
            raise AssertionError("QM.getDefinition() not implemented for: " + str(qm))
        raise ValueError("%s not for %s" % (qm, TargetType.SINGLE_NOMINAL))

    @staticmethod
    def required_stats(qm):
        r = _REQUIRED_STATS.get(qm)
        if r is not None:
            return r
        if qm.TARGET_TYPE in (TargetType.SINGLE_NUMERIC, TargetType.SINGLE_ORDINAL):
            raise AssertionError("QM.requiredStats() not implemented for: " + str(qm))
        raise ValueError("%s not for %s or %s" % (qm, TargetType.SINGLE_NUMERIC, TargetType.SINGLE_ORDINAL))


_DEFINITIONS = {
    QM.CORTANA_QUALITY: "WRAcc(x,y) / ((p/N)-((p/N)*(p/N)))",
    QM.WRACC: "(pos(y)/N)-(p/N)*(aCountBody(x,y)/N)",
    QM.MUTUAL_INFORMATION:
        " mi( (pos(y)/N)                        , (aTotalTargetCoverageNotBody(y)/N), (aCountNotHeadBody(x,y)/N)         ) +"
        " mi( (aCountNotHeadBody(x,y)/N)        , (aCountNotHeadNotBody(x)/N)       , (pos(y)/N)                         ) +"
        " mi( (aTotalTargetCoverageNotBody(y)/N), (pos(y)/N)                        , (aCountNotHeadNotBody(x)/N)        ) +"
        " mi( (aCountNotHeadNotBody(x)/N)       , (aCountNotHeadBody(x,y)/N)        , (aTotalTargetCoverageNotBody(y)/N) )",
    QM.ABSWRACC: "abs(WRAcc(x,y))",
    QM.CHI_SQUARED:
        " sqr(pos(y) - e11(N,aCountBody(x,y),p)) / e11(N,aCountBody(x,y),p) +"
        " sqr(p - pos(y) - e01(N,aCountBody(x,y),p)) / e01(N,aCountBody(x,y),p) +"
        " sqr(aCountBody(x,y) - pos(y) - e10(N,aCountBody(x,y),p)) / e10(N,aCountBody(x,y),p) +"
        " sqr(N - p - aCountBody(x,y) + pos(y) - e00(N,aCountBody(x,y),p)) / e00(N,aCountBody(x,y),p)",
    QM.INFORMATION_GAIN: "1 - 0.5*(x+y)*H(x/(x+y)) - 0.5*(2-x-y)*H((1-x)/(2-x-y))",
    QM.BINOMIAL: "sqrt(aCountBody(x,y)/N) * (pos(y)/aCountBody(x,y) - p/N)",
    QM.TARGET_SHARE: "pos(y)/aCountBody(x,y)",
    QM.PURITY: "max(pos(y)/aCountBody(x,y), 1.0-pos(y)/aCountBody(x,y))",
    QM.JACCARD: "pos(y)/(aCountBody(x,y) + aTotalTargetCoverageNotBody(y))",
    QM.COVERAGE: "aCountBody(x,y)",
    QM.SPECIFICITY: "aCountNotHeadNotBody(x)/(N - p)",
    QM.SENSITIVITY: "pos(y)/p",
    QM.LAPLACE: "(pos(y)+1)/(aCountBody(x,y)+2)",
    QM.F_MEASURE: "pos(y)/(p+aCountBody(x,y))",
    QM.G_MEASURE: "pos(y)/(aCountNotHeadBody(x,y)+p)",
    QM.CORRELATION: "(pos(y)*n - p*aCountNotHeadBody(x,y)) / sqrt(p*n*aCountBody(x,y)*(N-aCountBody(x,y)))",
    # TODO MM
    QM.PROP_SCORE_WRACC: "1/0 # TODO",
    QM.PROP_SCORE_RATIO: "1/0 # TODO",
    QM.BAYESIAN_SCORE: "1/0 # TODO",
    QM.LIFT: "(pos(y) * N) / (aCountBody(x,y) * p)",
    QM.RELATIVE_LIFT: "(pos(y) * N) / (aCountBody(x,y) * p) - 1",
}

_SUM = frozenset({Stat.SUM})
_SUM_SSD = frozenset({Stat.SUM, Stat.SSD})
_SUM_SSD_COMPL = frozenset({Stat.SUM, Stat.SSD, Stat.COMPL})
_MEDIAN_MAD = frozenset({Stat.MEDIAN, Stat.MAD})
_PDF = frozenset({Stat.PDF})

_REQUIRED_STATS = {
    # SINGLE_NUMERIC
    QM.Z_SCORE: _SUM, QM.INVERSE_Z_SCORE: _SUM, QM.ABS_Z_SCORE: _SUM,
    QM.AVERAGE: _SUM, QM.INVERSE_AVERAGE: _SUM, QM.QM_SUM: _SUM, QM.INVERSE_SUM: _SUM,
    QM.ABS_DEVIATION: _SUM, QM.MEAN_TEST: _SUM, QM.INVERSE_MEAN_TEST: _SUM, QM.ABS_MEAN_TEST: _SUM,
    QM.T_TEST: _SUM_SSD, QM.INVERSE_T_TEST: _SUM_SSD, QM.ABS_T_TEST: _SUM_SSD,
    QM.EXPLAINED_VARIANCE: _SUM_SSD_COMPL,
    QM.SQUARED_HELLINGER: _PDF, QM.SQUARED_HELLINGER_WEIGHTED: _PDF,
    QM.SQUARED_HELLINGER_WEIGHTED_ADJUSTED: _PDF, QM.KULLBACK_LEIBLER: _PDF,
    QM.KULLBACK_LEIBLER_WEIGHTED: _PDF, QM.CWRACC: _PDF, QM.CWRACC_UNWEIGHTED: _PDF,
    # SINGLE_ORDINAL
    QM.AUC: _SUM, QM.WMW_RANKS: _SUM, QM.INVERSE_WMW_RANKS: _SUM, QM.ABS_WMW_RANKS: _SUM,
    QM.MMAD: _MEDIAN_MAD,
}

# same self check as the static block in the java enum
for _qm in QM:
    if _qm.TARGET_TYPE == TargetType.SINGLE_NOMINAL:
        QM.get_definition(_qm)
    if _qm.TARGET_TYPE in (TargetType.SINGLE_NUMERIC, TargetType.SINGLE_ORDINAL):
        QM.required_stats(_qm)
