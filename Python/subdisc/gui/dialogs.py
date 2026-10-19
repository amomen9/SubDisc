"""The small modal dialogs: RandomQualitiesWindow, ConditionWindow, the
BasicJListWindow family (multi targets, remove domain), the CUI choosers."""

import re

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QDialog, QFormLayout, QGridLayout,
                               QHBoxLayout, QLabel, QListWidget, QMessageBox, QRadioButton,
                               QScrollArea, QVBoxLayout, QWidget)

from .. import log
from ..attribute_type import AttributeType
from ..condition import Condition
from ..condition_base import ConditionBase
from ..java_compat import float_to_string, parse_float
from ..operators import Operator
from ..validation import (DEFAULT_AMOUNT, RANDOM_DESCRIPTIONS, RANDOM_SUBSETS, SWAP_RANDOMIZATION,
                          Validation)
from . import common


def _dialog(parent, title):
    d = QDialog(parent)
    d.setWindowTitle(title)
    d.setWindowIcon(common.icon())
    d.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
    d.setModal(True)
    return d


_NUMBER_RE = re.compile(r"^\s*([+-]?)(\d[\d,]*)?(\.\d*)?")


def parse_number(text):
    """NumberFormat.getNumberInstance(Locale.US).parse(): reads the number at
    the start of the text, ignores the rest, raises ValueError when there is
    no number at all"""
    m = _NUMBER_RE.match(text or "")
    if m is None or (m.group(2) is None and (m.group(3) is None or len(m.group(3)) < 2)):
        raise ValueError('Unparseable number: "%s"' % text)
    s = m.group(1) + (m.group(2) or "0").replace(",", "") + (m.group(3) or "")
    return float(s)


# ----------------------------------------------------------------------
# RandomQualitiesWindow
# ----------------------------------------------------------------------

class RandomQualitiesWindow(QDialog):
    def __init__(self, target_type, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Which method?")
        self.setWindowIcon(common.icon())
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._settings = [None, None]
        valid = Validation.is_valid_random_qualities_target_type(target_type)
        lay = QVBoxLayout(self)
        box = common.build_border("Method to compute random qualities")
        bl = QVBoxLayout(box)
        self._methods = QButtonGroup(self)
        for i, m in enumerate((RANDOM_SUBSETS, RANDOM_DESCRIPTIONS, SWAP_RANDOMIZATION)):
            rb = QRadioButton(m)
            rb.setProperty("method", m)
            if i < 2:
                rb.setEnabled(valid)
            self._methods.addButton(rb)
            bl.addWidget(rb)
            if i == 2:
                rb.setChecked(True)
        row = QHBoxLayout()
        row.addWidget(QLabel("Amount"))
        row.addSpacing(50)
        self._amount = common.build_text_field(DEFAULT_AMOUNT)
        row.addWidget(self._amount)
        bl.addLayout(row)
        lay.addWidget(box)
        ok = common.build_button("OK", "O", self._ok, size=common.BUTTON_MEDIUM_SIZE)
        br = QHBoxLayout()
        br.addWidget(ok)
        br.addWidget(common.build_button("Cancel", "C", self.reject, size=common.BUTTON_MEDIUM_SIZE))
        br.addStretch(1)
        lay.addLayout(br)
        ok.setDefault(True)
        ok.setFocus()
        self.move(100, 100)

    def _ok(self):
        try:
            if int(parse_number(self._amount.text())) <= 1:
                QMessageBox.critical(self, "Invalid input", "Amount must be > 1.")
            else:
                self._settings = [self._methods.checkedButton().property("method"), self._amount.text()]
                self.accept()
        except ValueError as e:
            QMessageBox.critical(self, "Invalid input", str(e))

    def get_settings(self):
        """[method, amount], both None when cancelled"""
        self.exec()
        return self._settings


# ----------------------------------------------------------------------
# ConditionWindow
# ----------------------------------------------------------------------

class ConditionWindow(QDialog):
    """'Select Subset': one condition, Delete (or closing) removes the subset"""

    def __init__(self, parent, columns):
        super().__init__(parent)
        self.setWindowTitle("Specify Condition")
        self.setWindowIcon(common.icon())
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._columns = list(columns)
        self._condition = None
        lay = QVBoxLayout(self)
        box = common.build_border("Subset condition")
        form = QFormLayout(box)
        self._attribute = common.build_combo_box([c.get_name() for c in self._columns], self._update_operator_value)
        self._operator = common.build_combo_box()
        self._value_text = common.build_text_field("0")
        self._value_box = common.build_combo_box()
        form.addRow("attribute", self._attribute)
        form.addRow("operator", self._operator)
        self._value_row = QWidget()
        vl = QHBoxLayout(self._value_row)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.addWidget(self._value_text)
        vl.addWidget(self._value_box)
        form.addRow("value", self._value_row)
        lay.addWidget(box)
        br = QHBoxLayout()
        br.addStretch(1)
        sel = common.build_button("Select", "S", self._select)
        br.addWidget(sel)
        br.addWidget(common.build_button("Delete", "D", self._delete))
        br.addStretch(1)
        lay.addLayout(br)
        sel.setDefault(True)
        self._update_operator_value()
        self.move(200, 200)

    def _column(self):
        c = self._columns[0]
        name = self._attribute.currentText()
        for x in self._columns:
            if x.get_name() == name:
                c = x
        return c

    def _update_operator_value(self):
        self._operator.clear()
        self._value_box.clear()
        c = self._column()
        for o in Operator.get_operators(c.get_type()):
            if o.is_simple():
                self._operator.addItem(str(o))
        t = c.get_type()
        if t == AttributeType.NOMINAL:
            for v in c.get_domain():
                self._value_box.addItem(v)
        elif t == AttributeType.BINARY:
            # java shows an empty box here, which can not work
            self._value_box.addItems(["0", "1"])
        self._value_text.setVisible(t not in (AttributeType.NOMINAL, AttributeType.BINARY))
        self._value_box.setVisible(t in (AttributeType.NOMINAL, AttributeType.BINARY))

    def _delete(self):
        self._condition = None   # tells the parent Delete was used
        self.accept()

    def _select(self):
        c = self._column()
        op = Operator.from_string(self._operator.currentText())
        cb = ConditionBase(c, op)
        t = c.get_type()
        if t == AttributeType.NUMERIC:
            try:
                v = parse_float(self._value_text.text())
            except ValueError as e:
                QMessageBox.critical(self, "Invalid input", str(e))
                return
            n = c.size()
            c.build_sorted(np.zeros(n, dtype=bool))   # empty target, irrelevant here
            i = c.get_sorted_index(v)
            print("float: " + float_to_string(v) + ", index: " + str(i))
            self._condition = Condition(cb, v, i)
        elif t == AttributeType.BINARY:
            self._condition = Condition(cb, self._value_box.currentText() == "1")
        else:
            self._condition = Condition(cb, self._value_box.currentText())
        print(str(self._condition))
        self.accept()

    def get_result(self):
        self.exec()
        return self._condition


# ----------------------------------------------------------------------
# JList windows
# ----------------------------------------------------------------------

class SelectionList:
    """stands in for a swing JList that lives in the MiningWindow: the names
    and which ones are selected"""

    def __init__(self, items=()):
        self.items = list(items)
        self.selected = []

    def size(self):
        return len(self.items)

    def clear(self):
        self.items = []
        self.selected = []

    def add(self, item):
        self.items.append(item)

    def set_selection_interval(self, a, b):
        self.selected = list(range(a, b + 1)) if b >= a else []

    def clear_selection(self):
        self.selected = []

    def get_selected_indices(self):
        return sorted(self.selected)

    def get_selected_values(self):
        return [self.items[i] for i in self.get_selected_indices()]

    def is_selected_index(self, i):
        return i in self.selected


class BasicJListWindow(QDialog):
    def __init__(self, jlist, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowIcon(common.icon())
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._jlist = jlist
        self._original = jlist.get_selected_indices()
        lay = QVBoxLayout(self)
        self._top = QVBoxLayout()
        lay.addLayout(self._top)
        box = common.build_border("Select")
        bl = QVBoxLayout(box)
        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._list.addItems(jlist.items)
        for i in jlist.get_selected_indices():
            self._list.item(i).setSelected(True)
        self._list.itemSelectionChanged.connect(self._feedback_update)
        bl.addWidget(self._list, 1)
        sr = QHBoxLayout()
        sr.addWidget(common.build_button("Select All", "A", self._list.selectAll))
        sr.addWidget(common.build_button("Select None", "N", self._list.clearSelection))
        sr.addWidget(common.build_button("Invert Selection", "I", self._invert))
        bl.addLayout(sr)
        cr = QHBoxLayout()
        cr.addStretch(1)
        ok = common.build_button("OK", "O", self.dispose_ok)
        cr.addWidget(ok)
        cr.addWidget(common.build_button("Cancel", "C", self.dispose_cancel))
        cr.addStretch(1)
        bl.addLayout(cr)
        lay.addWidget(box, 1)
        self._feedback = QLabel()
        lay.addWidget(self._feedback)
        self._feedback_update()
        ok.setDefault(True)
        ok.setFocus()
        self.move(100, 100)

    def _feedback_update(self):
        self._feedback.setText(" %d of %d selected" % (len(self._list.selectedIndexes()), self._jlist.size()))

    def _invert(self):
        for i in range(self._list.count()):
            it = self._list.item(i)
            it.setSelected(not it.isSelected())

    def _selected(self):
        return sorted(self._list.row(it) for it in self._list.selectedItems())

    def _commit(self):
        self._jlist.selected = self._selected()

    def dispose_ok(self):
        self._commit()
        self.accept()

    def dispose_cancel(self):
        self.reject()

    def reject(self):
        # Cancel, Escape and the close button all end up here
        self._on_cancel()
        super().reject()

    def _on_cancel(self):
        self._jlist.clear_selection()


class _RestoringJListWindow(BasicJListWindow):
    """cancel puts the original selection back (multi target windows)"""

    def _on_cancel(self):
        self._jlist.selected = list(self._original)


class MultiTargetsWindow(_RestoringJListWindow):
    ERROR = ("Fields with invalid input: %d.\n"
             "Alpha and Beta must be >= 0.\nNumber of repetitions must be > 1.")

    def __init__(self, jlist, search_parameters, parent=None):
        super().__init__(jlist, "Multi-label details", parent)
        self._sp = search_parameters
        box = common.build_border("Settings")
        g = QGridLayout(box)
        self._alpha = common.build_text_field(float_to_string(search_parameters.get_alpha()))
        self._beta = common.build_text_field(float_to_string(search_parameters.get_beta()))
        self._repeated = common.build_combo_box(["No", "Yes"])
        self._repeated.setCurrentIndex(1)
        self._repetitions = common.build_text_field(str(search_parameters.get_post_processing_count()))
        for r, (lbl, w) in enumerate((("Alpha", self._alpha), ("Beta", self._beta),
                                      ("Repeated modeling", self._repeated),
                                      ("Number of repetitions", self._repetitions))):
            g.addWidget(QLabel(lbl), r, 0)
            g.addWidget(w, r, 1)
        self._top.addWidget(box)

    def dispose_ok(self):
        try:
            invalid = 0
            alpha = parse_number(self._alpha.text())
            if alpha < 0.0:
                self._alpha.selectAll()
                invalid += 1
            else:
                self._sp.set_alpha(alpha)
            beta = parse_number(self._beta.text())
            if beta < 0.0:
                self._beta.selectAll()
                invalid += 1
            else:
                self._sp.set_beta(beta)
            self._sp.set_post_processing_do_auto_run(self._repeated.currentIndex() == 1)
            reps = int(parse_number(self._repetitions.text()))
            if reps <= 1:
                self._repetitions.selectAll()
                invalid += 1
            else:
                self._sp.set_post_processing_count(reps)
            nr_selected = len(self._selected())
            if invalid == 0 and nr_selected > 0:
                self._commit()
                self.accept()
            # can both be true
            if invalid != 0:
                QMessageBox.critical(self, "Invalid input", self.ERROR % invalid)
            if nr_selected == 0:
                QMessageBox.critical(self, "Invalid input", "At least 1 Column must be selected.")
        except ValueError as e:
            QMessageBox.critical(self, "Invalid input", str(e))


class MultiNumericTargetsWindow(_RestoringJListWindow):
    def __init__(self, jlist, parent=None):
        super().__init__(jlist, "Multi-numeric targets", parent)

    def dispose_ok(self):
        # FIXME MM (java) maybe 1 is a valid setting (SINGLE_NUMERIC)
        if len(self._selected()) <= 1:
            QMessageBox.critical(self, "Invalid input", "At least 2 Columns must be selected.")
        else:
            self._commit()
            self.accept()


class RemoveDomainWindow(BasicJListWindow):
    def __init__(self, jlist, parent=None):
        super().__init__(jlist, "Remove Enrichment Source", parent)


def show_list_window(cls, jlist, *args, name=None):
    """the java constructors log and return when the list is None / empty"""
    if jlist is None:
        log.log_command_line((name or cls.__name__) + " Constructor: parameter can not be 'null'.")
        return False
    if jlist.size() == 0:
        log.log_command_line((name or cls.__name__) + " Constructor: the list can not be empty.")
        return False
    cls(jlist, *args).exec()
    return True


# ----------------------------------------------------------------------
# CUI choosers
# ----------------------------------------------------------------------

class CuiDomainChooser(QDialog):
    def __init__(self, domains, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CUI Domain Chooser")
        self.setWindowIcon(common.icon())
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._file = None
        lay = QVBoxLayout(self)
        box = common.build_border("Select CUI Domain")
        bl = QVBoxLayout(box)
        self._group = QButtonGroup(self)
        import os
        if not domains:
            bl.addWidget(QLabel("No Domain Files Found"))
        for i, f in enumerate(domains):
            name = os.path.basename(f)
            k = name.rfind(".")
            rb = QRadioButton(name if k < 0 else name[:k])
            rb.setProperty("path", f)
            self._group.addButton(rb)
            bl.addWidget(rb)
            if i == 0:
                rb.setChecked(True)
        lay.addWidget(box)
        if domains:
            br = QHBoxLayout()
            use = common.build_button("Use Domain", "U", self._use)
            br.addWidget(use)
            br.addWidget(common.build_button("Cancel", "C", self.reject))
            br.addStretch(1)
            lay.addLayout(br)
            use.setFocus()
        self.move(100, 100)

    def _use(self):
        b = self._group.checkedButton()
        self._file = b.property("path") if b is not None else None
        self.accept()

    def get_file(self):
        self.exec()
        return self._file


class IdentifierChooser(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Identifier Chooser")
        self.setWindowIcon(common.icon())
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._index = -1
        self._type = "entrez"
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Select the identifier column:"))
        self._cols = QButtonGroup(self)
        inner = QWidget()
        il = QVBoxLayout(inner)
        if not columns:
            il.addWidget(QLabel("No Column Headers Found"))
        for i, c in enumerate(columns):
            rb = QRadioButton(c.get_name())
            rb.setProperty("index", i)
            self._cols.addButton(rb)
            il.addWidget(rb)
            if i == 0:
                rb.setChecked(True)
        sc = QScrollArea()
        sc.setWidget(inner)
        sc.setWidgetResizable(True)
        lay.addWidget(sc, 1)
        lay.addSpacing(10)
        lay.addWidget(QLabel("Select the identifier type:"))
        tr = QHBoxLayout()
        self._types = QButtonGroup(self)
        for i, (text, key) in enumerate((("Entrez", "entrez"), ("GO", "go"))):
            rb = QRadioButton(text)
            rb.setProperty("key", key)
            self._types.addButton(rb)
            tr.addWidget(rb)
            if i == 0:
                rb.setChecked(True)
        lay.addLayout(tr)
        if columns:
            br = QHBoxLayout()
            br.addWidget(common.build_button("Use Column", "U", self._use))
            br.addWidget(common.build_button("Cancel", "C", self.reject))
            lay.addLayout(br)
        self.move(100, 100)

    def _use(self):
        self._index = self._cols.checkedButton().property("index")
        self._type = self._types.checkedButton().property("key")
        self.accept()

    def choose(self):
        self.exec()
        if self._index == -1:
            log.log_command_line("No identifier column selected.")
            return None
        return self._index, self._type
