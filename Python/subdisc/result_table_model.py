"""Column names and cell values of the result table (gui free part of the
java ResultTableModel; XMLAutoRun.save() uses the names too)."""

import math

from .number_format import format_number
from .java_compat import float_to_string
from .target_type import TargetType

COLUMN_COUNT = 8

_COL4 = {
    TargetType.SINGLE_NOMINAL: "Target Share",
    TargetType.SINGLE_NUMERIC: "Average",
    TargetType.MULTI_NUMERIC: "|Subgroup|",
    TargetType.DOUBLE_REGRESSION: "Slope",
    TargetType.DOUBLE_CORRELATION: "Correlation",
    TargetType.DOUBLE_BINARY: "Sign",
    TargetType.SCAPE: "Positives",
    TargetType.MULTI_LABEL: "Edit Distance",
    TargetType.LABEL_RANKING: "Ranking",
}

_COL5 = {
    TargetType.SINGLE_NOMINAL: "Positives",
    TargetType.SINGLE_NUMERIC: "St. Dev.",
    TargetType.MULTI_NUMERIC: "|!Subgroup|",
    TargetType.DOUBLE_REGRESSION: "Intercept",
    TargetType.DOUBLE_CORRELATION: "Distance",
    TargetType.DOUBLE_BINARY: "Target share dataset",
    TargetType.SCAPE: "Negatives",
    TargetType.MULTI_LABEL: "Entropy",
    TargetType.LABEL_RANKING: "",
}


def get_column_name(index, target_type):
    if index == 0:
        return "Nr."
    if index == 1:
        return "Depth"
    if index == 2:
        return "Coverage"
    if index == 3:
        return "Quality"
    if index == 4:
        if target_type not in _COL4:
            raise AssertionError(target_type)
        return _COL4[target_type]
    if index == 5:
        if target_type not in _COL5:
            raise AssertionError(target_type)
        return _COL5[target_type]
    if index == 6:
        return "p-Value"
    if index == 7:
        return "Conditions"
    raise AssertionError(index)


def get_value(subgroup, column, target_type):
    """the text the java table shows for a cell"""
    if column == 0:
        return subgroup.get_id()
    if column == 1:
        return subgroup.get_depth()
    if column == 2:
        return subgroup.get_coverage()
    if column == 3:
        return format_number(subgroup.get_measure_value())
    if column == 4:
        if target_type == TargetType.LABEL_RANKING:
            lr = subgroup.get_label_ranking()
            return lr.get_ranking() if lr is not None else ""
        return format_number(subgroup.get_secondary_statistic())
    if column == 5:
        return format_number(subgroup.get_tertiary_statistic())
    if column == 6:
        p = subgroup.get_p_value()
        return "  -" if math.isnan(p) else float_to_string(p)
    if column == 7:
        return str(subgroup.get_conditions())
    return "---"
