"""ResultWindow: the subgroups found, with Show Model, Browse, Delete,
Pattern Team, ROC, Save, Print and the p-value buttons. Also the
PatternTeamTabWindow (one tab per pattern team group)."""

import os
import time

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QKeySequence, QShortcut, QTextDocument
from PySide6.QtWidgets import (QAbstractItemView, QInputDialog, QMessageBox,
                               QTableView, QTabWidget)

from .. import log
from .. import probability_density_function as pdfm
from .. import result_table_model as rtm
from .. import ui_hooks
from ..file_type import FileType
from ..java_compat import double_to_string
from ..normal_distribution import NormalDistribution
from ..probability_density_function import ProbabilityDensityFunction, ProbabilityDensityFunction2
from ..regression_measure import RegressionMeasure
from ..target_type import TargetType
from ..validation import Validation, is_valid_random_qualities_setup
from ..xml_auto_run import save as xml_save
from . import common
from .browse_window import BrowseWindow
from .cauc_window import PLOT_CODE_PDF_2D, PLOT_CODE_PDF_2D_HEATMAP
from .dialogs import RandomQualitiesWindow
from .label_ranking_matrix_window import LabelRankingMatrixWindow
from .model_window import ModelWindow
from .pdf_window_2d import PDFWindow2D
from .roc_curve_window import ROCCurveWindow


def create_title(subgroup):
    return "Subgroup %d (%s)" % (subgroup.get_id(), subgroup)


class ResultTableModel(QAbstractTableModel):
    def __init__(self, subgroup_set, target_type):
        super().__init__()
        self._set = subgroup_set
        self._tt = target_type
        self._items = list(subgroup_set)

    def items(self):
        return self._items

    def refresh(self):
        self.beginResetModel()
        self._items = list(self._set)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._items)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else rtm.COLUMN_COUNT

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return rtm.get_column_name(section, self._tt)
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            v = rtm.get_value(self._items[index.row()], index.column(), self._tt)
            return str(v)
        if role == Qt.TextAlignmentRole and index.column() in (0, 1, 2, 3, 5, 6):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.TextAlignmentRole and index.column() == 4 and self._tt != TargetType.LABEL_RANKING:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        return None


def _result_view(model):
    v = QTableView()
    v.setModel(model)
    v.setSelectionBehavior(QAbstractItemView.SelectRows)
    v.setSelectionMode(QAbstractItemView.ExtendedSelection)
    v.setAlternatingRowColors(True)
    v.verticalHeader().setVisible(False)
    v.verticalHeader().setDefaultSectionSize(22)
    # 8 columns, 20 unit widths
    unit = int(0.05 * common.WINDOW_DEFAULT_SIZE.width())
    for c, f in enumerate((1, 1, 1.75, 1.75, 1.75, 1.75, 2, 9)):
        v.setColumnWidth(c, int(f * unit))
    v.horizontalHeader().setStretchLastSection(True)
    # sorting stays off, it would destroy the order of the subgroups
    return v


class ResultWindow(common.Window):
    def __init__(self, table, selection, subgroup_discovery, subgroup_set=None):
        super().__init__()
        if table is None or subgroup_discovery is None:
            log.log_command_line("ResultWindow Constructor: parameter(s) can not be 'null'.")
            return
        self._table = table
        self._selection = selection
        self._sd = subgroup_discovery
        self._set = subgroup_discovery.get_result() if subgroup_set is None else subgroup_set
        self._sp = subgroup_discovery.get_search_parameters()
        self._qm = subgroup_discovery.get_quality_measure()
        # only for DOUBLE_REGRESSION, avoids recalculation
        self._rm_base = subgroup_discovery.get_regression_measure_base()
        tt = self._sp.get_target_concept().get_target_type()
        self._model = ResultTableModel(self._set, tt)
        self._view = _result_view(self._model)
        self.content().addWidget(self._view, 1)
        if len(self._model.items()) > 0:
            self._view.selectRow(0)
        self._view.doubleClicked.connect(lambda _i: self._enter())

        self._show_model = common.build_button("Show Model", "M", self._show_model_clicked)
        self._browse = common.build_button("Browse", "B", self._browse_clicked)
        self._delete = common.build_button("Delete", "D", self._delete_clicked)
        self.add_button_row([self._show_model, self._browse, self._delete])
        self._pattern_team = common.build_button("Pattern Team", "P", self._pattern_team_clicked)
        self._roc = common.build_button("ROC", "R", self._roc_clicked, common.BUTTON_MEDIUM_SIZE)
        self._save = common.build_button("Save", "S", self._save_clicked, common.BUTTON_MEDIUM_SIZE)
        self._print = common.build_button("Print", "I", self._print_clicked, common.BUTTON_MEDIUM_SIZE)
        self._p_values = common.build_button("Gaussian p-Values", "V", self._p_values_clicked)
        self._regression = common.build_button("Regression Test", "T", self._regression_clicked)
        self._empirical = common.build_button("Empirical p-Values", "E", self._empirical_clicked)
        self.add_button_row([self._pattern_team, self._roc, self._save, self._print,
                             self._p_values, self._regression, self._empirical])
        self.add_button_row([common.build_button("Close", "C", self.close)])
        self._enable_buttons_check()

        QShortcut(QKeySequence(Qt.Key_Return), self._view, self._enter, context=Qt.WidgetShortcut)
        QShortcut(QKeySequence(Qt.Key_Enter), self._view, self._enter, context=Qt.WidgetShortcut)
        QShortcut(QKeySequence(Qt.Key_Delete), self._view,
                  lambda: self._model.rowCount() > 0 and self._delete_clicked(), context=Qt.WidgetShortcut)

        self._set_title()
        self.show_at(200, 200, common.WINDOW_DEFAULT_SIZE)

    @classmethod
    def pattern_team_window(cls, parent, pattern_team):
        return cls(parent._table, parent._selection, parent._sd, pattern_team)

    def _enter(self):
        if self._model.rowCount() > 0 and self._sp.get_target_type() != TargetType.SINGLE_NOMINAL:
            self._show_model_clicked()

    def _set_title(self):
        s = []
        if len(self._set) == 0:
            s.append("No subgroups found that match the set criterion")
        else:
            s.append("%d subgroups found" % len(self._set))
        tt = self._sp.get_target_type()
        tc = self._sp.get_target_concept()
        if tt in (TargetType.SINGLE_NOMINAL, TargetType.SINGLE_NUMERIC):
            s.append("; target = " + tc.get_primary_target().get_name())
        if TargetType.has_target_value(tt):
            s.append("; value = ")
            if tt == TargetType.SINGLE_NOMINAL:
                s.append(str(tc.get_target_value()))
            else:
                s.append(tc.get_secondary_target().get_name())
        s.append("; quality measure = " + self._sp.get_quality_measure().GUI_TEXT)
        je = self._set.get_joint_entropy()
        if je == je:
            s.append("; joint entropy = " + common.number(je, 3))
        self.setWindowTitle("".join(s))

    def _enable_buttons_check(self):
        if len(self._set) == 0:
            for b in (self._show_model, self._roc, self._pattern_team, self._browse, self._delete,
                      self._p_values, self._regression, self._empirical, self._save, self._print):
                b.setEnabled(False)
            return
        tt = self._sp.get_target_type()
        self._show_model.setVisible(TargetType.has_base_model(tt))
        self._roc.setVisible(tt == TargetType.SINGLE_NOMINAL)

    def _selected_rows(self):
        return sorted({i.row() for i in self._view.selectionModel().selectedRows()})

    def _selected_subgroups(self):
        items = self._model.items()
        return [items[r] for r in self._selected_rows()]

    # ------------------------------------------------------------------
    # show model
    # ------------------------------------------------------------------

    def _show_model_clicked(self):
        sel = self._selected_subgroups()
        if not sel:
            return
        tt = self._sp.get_target_type()
        try:
            if tt == TargetType.SINGLE_NUMERIC:
                self._show_single_numeric(sel)
            elif tt == TargetType.MULTI_NUMERIC:
                self._show_multi_numeric(sel)
            elif tt in (TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION):
                self._show_double(tt, sel)
            elif tt == TargetType.SCAPE:
                self._show_scape(sel)
            elif tt == TargetType.MULTI_LABEL:
                self._show_multi_label()
            elif tt == TargetType.LABEL_RANKING:
                self._show_label_ranking(sel)
            else:
                raise AssertionError(tt)
        finally:
            common.set_busy(False)

    def _base_pdf(self, column):
        if not pdfm.USE_ProbabilityDensityFunction2:
            pdf = ProbabilityDensityFunction(column)
        else:
            pdf = ProbabilityDensityFunction2(column, self._selection)
        pdf.smooth()
        return pdf

    @staticmethod
    def _sub_pdf(pdf, members):
        if not pdfm.USE_ProbabilityDensityFunction2:
            p = ProbabilityDensityFunction(pdf, members)
        else:
            p = ProbabilityDensityFunction2(pdf, members)
        p.smooth()
        return p

    def _show_single_numeric(self, sel):
        target = self._sp.get_target_concept().get_primary_target()
        pdf = self._base_pdf(target)
        for s in sel:
            ModelWindow.numeric(target, pdf, self._sub_pdf(pdf, s.get_members()), create_title(s), False)

    def _show_multi_numeric(self, sel):
        # FIXME MM (java) ShowModel only for 2D targets
        lst = self._sp.get_target_concept().get_multi_targets()
        if len(lst) != 2:
            QMessageBox.warning(self, "Show Model", "MULTI_NUMERIC base model only available for 2 dimensions")
            return
        pdf = self._sd.pdf_nd
        limits = pdf.get_limits()
        for s in sel:
            common.set_busy(True)
            grids = pdf.get_density_difference_2d(s.get_members(), True, None)
            stats = list(pdf.last_dxdy)
            common.set_busy(False)
            grids = [np.asarray(g, dtype=np.float32) for g in grids]
            norm = _normalise_grids(grids)
            PDFWindow2D.grids(grids, stats, limits, create_title(s), lst[1].get_name(), lst[0].get_name())
            _write_pdfs(s, grids, stats, norm, lst[0].get_name(), lst[1].get_name())

    def _show_double(self, tt, sel):
        tc = self._sp.get_target_concept()
        is_regression = tt == TargetType.DOUBLE_REGRESSION
        rm = None
        for s in sel:
            if is_regression:
                rm = RegressionMeasure.from_base(self._rm_base, s.get_members())
                rm.get_evaluation_measure_value()
            ModelWindow.scatter(self._table, tc.get_primary_target(), tc.get_secondary_target(), rm, s)

    def _show_scape(self, sel):
        tc = self._sp.get_target_concept()
        binary, numeric = tc.get_primary_target(), tc.get_secondary_target()
        binaries = np.asarray(binary.get_binaries(), dtype=bool)
        pdf = self._base_pdf(numeric)
        non_target = _flip_to_length(binaries)
        for s in sel:
            m = np.asarray(s.get_members(), dtype=bool)
            n = min(len(m), len(binaries))
            pos = np.zeros_like(m)
            pos[:n] = m[:n] & binaries[:n]
            neg = np.zeros_like(m)
            neg[:n] = m[:n] & non_target[:n]
            ModelWindow.numeric(numeric, self._sub_pdf(pdf, pos), self._sub_pdf(pdf, neg), create_title(s), True)

    def _show_multi_label(self):
        items = self._model.items()
        for r in self._selected_rows():
            log.log_command_line("subgroup " + str(r + 1))
            w = ModelWindow.bayesian_network(items[r].get_dag(), 1000, 800)
            w.setWindowTitle("Subgroup %d: induced Bayesian Network" % (r + 1))

    def _show_label_ranking(self, sel):
        for s in sel:
            lrm = s.get_label_ranking_matrix()
            lrm.print()
            LabelRankingMatrixWindow(self._qm.get_base_label_ranking_matrix(), lrm,
                                     create_title(s) + "   " + s.get_label_ranking().get_ranking())

    # ------------------------------------------------------------------
    # other buttons
    # ------------------------------------------------------------------

    def _roc_clicked(self):
        ROCCurveWindow(self._set, self._sp, self._qm)

    def _pattern_team_clicked(self):
        text, ok = QInputDialog.getText(self, "Input", "Set pattern team size:")
        if not ok or text == "":
            return
        common.set_busy(True)
        try:
            k = int(text.strip())
            team = self._set.get_pattern_team(self._table, k)
            ResultWindow.pattern_team_window(self, team)
            grouping = self._set.get_grouping(team)
            PatternTeamTabWindow(grouping, self._sp)
        except Exception as e:
            log.log_command_line(str(e))
            common.set_busy(False)
            QMessageBox.critical(self, "Warning", "Not a valid input value!")
        finally:
            common.set_busy(False)

    def _browse_clicked(self):
        for s in self._selected_subgroups():
            BrowseWindow(self._table, s)

    def _delete_clicked(self):
        if len(self._set) == 0:
            return
        for s in reversed(self._selected_subgroups()):
            self._set.remove(s)
        self._model.refresh()
        self._view.clearSelection()
        if len(self._set) > 0:
            self._view.selectRow(0)
        else:
            self._delete.setEnabled(False)

    def _random_qualities(self, then):
        """asks for the setup, computes the random qualities on a worker
        thread and calls then(qualities) when done"""
        setup = RandomQualitiesWindow(self._sp.get_target_type(), self).get_settings()
        log.log_command_line("%s=%s" % (setup[0] if setup[0] is not None else "null",
                                        setup[1] if setup[1] is not None else "null"))
        if not is_valid_random_qualities_setup(setup):
            return
        v = Validation(self._sp, self._table, self._selection, self._qm)
        common.BackgroundTask(self, "Computing random qualities...",
                              "Computing %s random qualities (%s)." % (setup[1], setup[0]),
                              lambda: v.get_qualities(setup),
                              lambda q: q is not None and then(q)).start()

    def _p_values_clicked(self):
        def done(q):
            d = NormalDistribution(q)
            for s in self._set:
                s.set_p_value(d)
            self._view.viewport().update()
        self._random_qualities(done)

    def _regression_clicked(self):
        def done(q):
            score = Validation.perform_regression_test(q, self._set)
            QMessageBox.information(self, "Message", "The regression test score equals\nfor k =  1 : %s\nfor k = 10 : %s"
                                    % (double_to_string(score[0]), double_to_string(score[1])))
        self._random_qualities(done)

    def _empirical_clicked(self):
        def done(q):
            p = Validation.compute_empirical_p_value(q, self._set)
            QMessageBox.information(self, "Message", "The empirical p-value is p = " + double_to_string(p))
            for s in self._set:
                s.set_empirical_p_value(q)
            self._view.viewport().update()
        self._random_qualities(done)

    def _save_clicked(self):
        path = ui_hooks.choose_save_file(FileType.ALL_DATA_FILES)
        if path is None:
            return   # cancelled
        xml_save(self._set, os.path.abspath(path), self._sp.get_target_type())

    def _print_clicked(self):
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter
        printer = QPrinter(QPrinter.HighResolution)
        dlg = QPrintDialog(printer, self)
        if dlg.exec() != QPrintDialog.Accepted:
            return
        tt = self._sp.get_target_type()
        rows = ["<tr>" + "".join("<th>%s</th>" % _esc(rtm.get_column_name(c, tt)) for c in range(rtm.COLUMN_COUNT)) + "</tr>"]
        for s in self._model.items():
            rows.append("<tr>" + "".join("<td>%s</td>" % _esc(str(rtm.get_value(s, c, tt))) for c in range(rtm.COLUMN_COUNT)) + "</tr>")
        doc = QTextDocument()
        doc.setHtml("<table border='1' cellspacing='0' cellpadding='2'>%s</table>" % "".join(rows))
        try:
            doc.print_(printer)
        except Exception as e:
            QMessageBox.warning(self, "Warning", "Print error!")
            log.log_command_line(str(e))


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _flip_to_length(bits):
    """BitSet.flip(0, length()): length is the highest set bit + 1, so the
    rows after the last positive stay 0 (as in java)"""
    out = np.zeros_like(bits)
    nz = np.flatnonzero(bits)
    if nz.size:
        L = int(nz[-1]) + 1
        out[:L] = ~bits[:L]
    return out


def _normalise_grids(grids):
    """divide all 3 grids (subgroup, complement, difference) by their common
    maximum, so they share the same scale"""
    mx = np.float32(-3.4028235e38)
    for g in grids:
        if g.size:
            mx = max(mx, np.float32(g.max()))
    with np.errstate(all="ignore"):
        for g in grids:
            g /= mx
    return float(mx)


def _write_pdfs(subgroup, grids, stats, normaliser, x_name, y_name):
    log.log_command_line("writing pdf grid files for: " + create_title(subgroup))
    # avoids overwriting + keeps the 3 grids together in file browsers
    base = "%d_%%s_%d.dat" % (time.time_ns(), subgroup.get_id())
    info = create_title(subgroup)
    for g, kind in zip(grids, ("subgroup", "complement", "difference")):
        name = base % kind
        _write_pdf(g, stats, normaliser, name, info)
        _write_gnuplot_script(name, x_name, y_name, False)
        _write_gnuplot_script(name, x_name, y_name, True)


def _write_pdf(grid, stats, normaliser, file_name, info):
    try:
        with open(file_name, "w", encoding="utf-8", newline="") as w:
            log.log_command_line("writing: " + os.path.abspath(file_name))
            x_min, y_min, dx, dy = stats[0], stats[3], stats[6], stats[7]
            w.write("# " + info + "\n")
            w.write("# stats [" + ", ".join(double_to_string(v) for v in stats) + "]\n")
            w.write("# normalisation " + double_to_string(normaliser) + "\n")
            w.write("# y\tx\tz\n")   # NOTICE SWAP
            for i in range(grid.shape[0]):
                x = x_min + dx * i
                for j in range(grid.shape[1]):
                    y = y_min + dy * j
                    w.write("%s\t%s\t%s\n" % (double_to_string(y), double_to_string(x), double_to_string(float(grid[i, j]))))
                w.write("\n")   # gnuplot needs this at a y change
            log.log_command_line("Done\n")
    except OSError as e:
        log.log_command_line(str(e))


def _write_gnuplot_script(data_file, x_name, y_name, heat_map):
    path = data_file + (".heatmap.gp" if heat_map else ".gp")
    try:
        with open(path, "w", encoding="utf-8", newline="") as w:
            log.log_command_line("writing: " + os.path.abspath(path))
            script = PLOT_CODE_PDF_2D_HEATMAP if heat_map else PLOT_CODE_PDF_2D
            w.write(script % (data_file, y_name, x_name))   # NOTICE SWAP OF X AND Y
            log.log_command_line("Done\n")
    except OSError as e:
        log.log_command_line(str(e))


class PatternTeamTabWindow(common.Window):
    def __init__(self, grouping, search_parameters):
        super().__init__("Grouping of Subgroups According to Pattern Teams")
        tabs = QTabWidget()
        tabs.setUsesScrollButtons(True)
        tt = search_parameters.get_target_concept().get_target_type()
        self._views = []
        for i, ss in enumerate(grouping, 1):
            v = _result_view(ResultTableModel(ss, tt))
            self._views.append(v)
            tabs.addTab(v, "Group %d (%d)" % (i, len(ss)))
        self.content().addWidget(tabs, 1)
        self.show_at(300, 300, common.WINDOW_DEFAULT_SIZE)
