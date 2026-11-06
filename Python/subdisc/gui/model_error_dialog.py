"""Tools menu of the main window: Random Holdout Split and Add Model Error
Columns. Not in java."""

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from .. import holdout
from .. import model_errors
from ..attribute_type import AttributeType
from . import common
from .feature_common import SettingsDialog, html_table


def ask_holdout_split(parent, table):
    """train mask or None; the test rows are the ones left out"""
    d = SettingsDialog(parent, "Random Holdout Split",
                       "Keeps a random part of the rows out of the mining run. Mining then uses only the other rows "
                       "(as a selection), and 'Holdout Test' / 'Subgroup List' in the result window can score the "
                       "result on the rows that were kept out. This replaces any selection made with Select Subset.")
    d.add_float("fraction", "Test rows (fraction)", 0.3, 0.05, 0.9, 2, 0.05)
    d.add_int("seed", "Random seed", 1, 0, 2 ** 31 - 1)
    v = d.values()
    if v is None:
        return None
    train, test = holdout.random_split(table.get_nr_rows(), v["fraction"], v["seed"])
    return train


def ask_model_errors(parent, table, default_target=None, default_value=None):
    """adds the model columns; returns (new names, target name, value) or None"""
    candidates = [c for c in table.get_columns() if c.get_type() in (AttributeType.NOMINAL, AttributeType.BINARY)]
    if not candidates:
        QMessageBox.information(parent, "Model Errors", "The table has no nominal or binary column to predict.")
        return None
    names = [c.get_name() for c in candidates]
    d = SettingsDialog(parent, "Add Model Error Columns",
                       "Trains a logistic regression with k-fold cross-validation (every row is predicted by a model "
                       "that did not see it) on all enabled columns, then adds score, error, false positive and false "
                       "negative columns. Mine with the error column as target to see where the model fails.")
    target_box = d.add_combo("target", "Column to predict", names, default_target)
    value_box = d.add_combo("value", "Positive value", [])

    def fill_values():
        value_box.clear()
        col = table.get_column(target_box.currentText())
        if col is None:
            return
        if col.get_type() == AttributeType.BINARY:
            value_box.addItems(["1", "0"])
        else:
            value_box.addItems(sorted(col.distinct_values()))
        if default_value is not None and value_box.findText(default_value) >= 0:
            value_box.setCurrentText(default_value)

    target_box.currentIndexChanged.connect(lambda _i: fill_values())
    fill_values()
    d.add_int("folds", "Folds", 5, 2, 50)
    d.add_float("l2", "L2 penalty (lambda)", 1.0, 0.0, 1e6, 3, 0.5)
    d.add_float("threshold", "Predict positive when score >=", 0.5, 0.01, 0.99, 2, 0.05)
    d.add_int("seed", "Random seed", 1, 0, 2 ** 31 - 1)
    d.add_check("disable", "Exclude the new columns and the predicted column from the search conditions", True)
    v = d.values()
    if v is None:
        return None
    target = table.get_column(v["target"])
    common.set_busy(True)
    try:
        new, info = model_errors.add_model_error_columns(table, target, v["value"], None, v["folds"], v["l2"],
                                                         v["threshold"], v["seed"])
    except Exception as e:
        common.set_busy(False)
        QMessageBox.warning(parent, "Model Errors", str(e))
        return None
    finally:
        common.set_busy(False)
    if v["disable"]:
        for name in new:
            table.get_column(name).set_is_enabled(False)
        target.set_is_enabled(False)
    box = QMessageBox(parent)
    box.setWindowTitle("Model Errors")
    box.setTextFormat(Qt.RichText)
    box.setText("Added %s.<br><br>%s<br>The target is now <b>%s = 1</b>; Subgroup Discovery finds where the "
                "model is wrong more often than on average.%s"
                % (", ".join(new), html_table(info), new[1],
                   "<br>The new columns and '%s' are switched off as search conditions (Meta Data can turn them on again)."
                   % target.get_name() if v["disable"] else ""))
    box.exec()
    return new, target.get_name(), v["value"]


def nr_rows_text(mask):
    return "{:,}".format(int(np.count_nonzero(mask)))
