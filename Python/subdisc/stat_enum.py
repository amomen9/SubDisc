from enum import Enum


class Stat(Enum):
    """Which statistics a SINGLE_NUMERIC quality measure needs.

    SUM: sum, SSD: sum of squared deviations, COMPL: complement, MEDIAN,
    MAD: median absolute deviation, PDF: probability density function.
    Computing them all every time wastes a lot of cpu for nothing.
    """
    SUM = 0      # MEAN_TEST, Z-SCORE, T_TEST, AVERAGE, AUC, WMW_RANKS
    SSD = 1      # T_TEST
    COMPL = 2    # EXPLAINED_VARIANCE
    MEDIAN = 3   # MMAD
    MAD = 4      # MMAD
    PDF = 5      # HELLINGER, KULLBACK_LEIBLER, CWRACC
