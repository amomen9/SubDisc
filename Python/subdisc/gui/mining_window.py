"""MiningWindow: the main window. Four panels (Dataset, Target Concept, Search
Conditions, Search Strategy), the mining buttons and the menus.

How a change flows down (same as java):

target type changed -> init quality measures -> init target attributes ->
show/hide fields -> target attribute changed -> init target values ->
misc field changed -> init target info

Swing combo boxes fire an ActionEvent on every programmatic change while a
listener is attached. Here the handlers are only connected to user actions
(QComboBox.activated) and the places where java relied on such an event
call the handler directly.
"""

import copy
import os
import traceback

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QHBoxLayout, QInputDialog,
                               QLabel, QMessageBox, QStatusBar, QVBoxLayout, QWidget)

from .. import log
from .. import probability_density_function as pdfm
from .. import process
from .. import ui_hooks
from ..attribute_type import AttributeType
from ..condition_list_builder import create_list
from ..correlation_measure import CorrelationMeasure
from ..cross_validation import CrossValidation
from ..cui import EnrichmentType
from ..file_handler import FileHandler
from ..file_type import FileType
from ..java_compat import float_to_string, double_to_string, parse_float, parse_int
from ..label_ranking_matrix import LabelRankingMatrix
from ..normal_distribution import NormalDistribution
from ..numeric_operator_setting import NumericOperatorSetting
from ..numeric_strategy import NumericStrategy
from ..probability_density_function import ProbabilityDensityFunction, ProbabilityDensityFunction2
from ..probability_density_function_2d import ProbabilityDensityFunction2D
from ..qm import QM
from ..quality_measure import QualityMeasure
from ..regression_measure import RegressionMeasure
from ..search_parameters import ALPHA_EDIT_DISTANCE, SearchParameters
from ..search_strategy import SearchStrategy
from ..stat_enum import Stat
from ..table import Table
from ..target_concept import TargetConcept
from ..target_type import TargetType
from ..validation import Validation, is_valid_random_qualities_setup
from ..xml_auto_run import AutoRun, XMLAutoRun, save as xml_save
from .. import __version__
from . import common
from . import theme
from .browse_window import BrowseWindow
from .dialogs import (ConditionWindow, MultiNumericTargetsWindow, MultiTargetsWindow, RandomQualitiesWindow,
                      RemoveDomainWindow, SelectionList, show_list_window)
from .histogram_window import HistogramWindow
from .label_ranking_matrix_window import LabelRankingMatrixWindow
from .meta_data_window import MetaDataWindow
from .model_window import BASE_MODEL_TEXT, ModelWindow
from .multi_regression_targets_window import MultiRegressionTargetsWindow
from .pdf_window_2d import PDFWindow2D

# leave False
ADD_DISCRETISE_BUTTON = False
TITLE = "SubDisc: Subgroup Discovery Tool"


class MiningWindow(common.Window):
    """MiningWindow(), MiningWindow(table) or MiningWindow(table, search_parameters)
    (the last one when loaded from xml)"""

    _title_signal = Signal(str)

    def __init__(self, table=None, search_parameters=None):
        super().__init__()
        self._table = table
        self._selection = None   # a subset of the data set by the user, None for all
        self._dbc = None
        self._positive_count = 0
        self._positive_count_selection = 0
        self._target_average = 0.0
        self._target_average_selection = 0.0
        self._search_parameters = SearchParameters()
        self._target_concept = TargetConcept()
        self._is_cancelled = False
        self._multi_targets = SelectionList()
        self._multi_regression_targets = SelectionList()
        self._title_signal.connect(self._set_title_now, Qt.QueuedConnection)

        self._init_components()
        self._init_static_gui_components()
        if table is not None and search_parameters is not None:
            self._target_concept = search_parameters.get_target_concept() or TargetConcept()
            self._search_parameters = search_parameters
            self._init_gui_components_from_file()
        elif table is not None:
            self._target_type_changed()
            self._init_gui_components()
        self._enable_table_dependent_components(self._table is not None)
        self.init_title()
        log.open_file_output_streams()
        self.move(100, 100)
        self.resize(940, 640)

    # ------------------------------------------------------------------
    # building the window
    # ------------------------------------------------------------------

    def _action(self, text, mnemonic, slot, accel=True):
        a = QAction(common.with_mnemonic(text, mnemonic), self)
        if accel:
            a.setShortcut(QKeySequence("Ctrl+" + mnemonic.upper()))
            a.setShortcutContext(Qt.WindowShortcut)
        a.triggered.connect(lambda _=False: slot())
        return a

    def _init_components(self):
        mb = self.menuBar()
        # File
        m = mb.addMenu(common.with_mnemonic("File", "F"))
        self._m_open = self._action("Open File...", "O", self._open_file)
        m.addAction(self._m_open)
        self._m_db = self._action("Load PSR Table", "Q", self._load_database_table)
        m.addAction(self._m_db)
        self._m_browse = self._action("Browse...", "B", self._browse)
        m.addAction(self._m_browse)
        self._m_explore = self._action("Explore...", "E", self._explore)
        m.addAction(self._m_explore)
        self._m_meta = self._action("Meta Data...", "D", self._meta_data)
        m.addAction(self._m_meta)
        self._m_subset = self._action("Select Subset...", "Y", self._select_subset)
        m.addAction(self._m_subset)
        m.addSeparator()
        self._m_sd = self._action("Subgroup Discovery", "S", self._subgroup_discovery)
        m.addAction(self._m_sd)
        m.addSeparator()
        self._m_create_autorun = self._action("Create Autorun File", "C", lambda: self._autorun_file(AutoRun.CREATE))
        m.addAction(self._m_create_autorun)
        self._m_add_autorun = self._action("Add to Autorun File", "A", lambda: self._autorun_file(AutoRun.ADD))
        m.addAction(self._m_add_autorun)
        self._m_sampled = self._action("Load Sampled Subgroups", "L", self._load_sampled_subgroups)
        m.addAction(self._m_sampled)
        m.addSeparator()
        m.addAction(self._action("Exit", "X", self._exit))
        # Enrichment
        e = mb.addMenu(common.with_mnemonic("Enrichment", "N"))
        self._m_cui = self._action("Add CUI Domain", "C", lambda: self._add_enrichment_source(EnrichmentType.CUI), False)
        e.addAction(self._m_cui)
        # TODO MM (java) add when implemented
        self._m_go = self._action("Add GO Domain", "G", lambda: self._add_enrichment_source(EnrichmentType.GO), False)
        self._m_custom = self._action("Add Custom Domain", "U", lambda: self._add_enrichment_source(EnrichmentType.CUSTOM), False)
        self._m_remove_domain = self._action("Remove Enrichment Source", "R", self._remove_enrichment_source, False)
        self._m_remove_domain.setEnabled(False)
        e.addAction(self._m_remove_domain)
        # View (not in java: light / dark look)
        v = mb.addMenu(common.with_mnemonic("View", "V"))
        grp = QActionGroup(self)
        for name, key in (("Light", theme.LIGHT), ("Dark", theme.DARK)):
            a = QAction(name, self, checkable=True)
            a.setChecked(theme.current() == key)
            a.triggered.connect(lambda _=False, k=key: theme.apply(name=k))
            grp.addAction(a)
            v.addAction(a)
        # About
        ab = mb.addMenu(common.with_mnemonic("About", "A"))
        ab.addAction(self._action("SubDisc", "I", self._about))

        center = QWidget()
        grid = QGridLayout(center)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(8)
        grid.addWidget(self._build_data_set_panel(), 0, 0)
        grid.addWidget(self._build_target_concept_panel(), 0, 1)
        grid.addWidget(self._build_search_conditions_panel(), 1, 0)
        grid.addWidget(self._build_search_strategy_panel(), 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.content().addWidget(center, 1)

        self._b_sd = common.build_button("Subgroup Discovery", "S", self._subgroup_discovery)
        self._b_sd_loop = common.build_button("SD Loop", "L", self._subgroup_discovery_loop)
        self._b_cv = common.build_button("Cross-Validate", "V", self._cross_validate)
        self._b_threshold = common.build_button("Compute Threshold", "P", self._compute_threshold)
        self._b_sd.setDefault(True)
        self.add_button_row([self._b_sd, self._b_sd_loop, self._b_cv, self._b_threshold])

        self._status = QStatusBar()
        self.setStatusBar(self._status)

    def _form(self, box):
        f = QFormLayout(box)
        f.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        f.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        f.setHorizontalSpacing(24)
        return f

    def _build_data_set_panel(self):
        box = common.build_border("Dataset")
        v = QVBoxLayout(box)
        f = QFormLayout()
        f.setHorizontalSpacing(40)
        self._l_table_name = QLabel("")
        self._l_nr_examples = QLabel("")
        self._l_nr_attributes = QLabel("")
        self._l_nr_nominals = QLabel("")
        self._l_nr_numerics = QLabel("")
        self._l_nr_binaries = QLabel("")
        for lbl, w in (("target table", self._l_table_name), ("# examples", self._l_nr_examples),
                       ("# attributes", self._l_nr_attributes), ("# nominals", self._l_nr_nominals),
                       ("# numerics", self._l_nr_numerics), ("# binaries", self._l_nr_binaries)):
            w.setTextInteractionFlags(Qt.TextSelectableByMouse)
            f.addRow(lbl, w)
        v.addLayout(f)
        v.addStretch(1)
        g = QGridLayout()
        self._b_browse = common.build_button("Browse...", "B", self._browse)
        self._b_explore = common.build_button("Explore...", "E", self._explore)
        self._b_meta = common.build_button("Meta Data...", "D", self._meta_data)
        self._b_subset = common.build_button("Select Subset...", "Y", self._select_subset)
        self._b_discretise = common.build_button("Discretise", "Z", self._discretise)
        g.addWidget(self._b_browse, 0, 0)
        g.addWidget(self._b_explore, 0, 1)
        g.addWidget(self._b_meta, 1, 0)
        g.addWidget(self._b_subset, 1, 1)
        if ADD_DISCRETISE_BUTTON:
            g.addWidget(self._b_discretise, 2, 0)
        h = QHBoxLayout()
        h.addStretch(1)
        h.addLayout(g)
        h.addStretch(1)
        v.addLayout(h)
        return box

    def _build_target_concept_panel(self):
        box = common.build_border("Target Concept")
        f = self._form(box)
        self._c_target_type = common.build_combo_box(slot=self._target_type_changed)
        self._c_quality_measure = common.build_combo_box(slot=self._quality_measure_changed)
        self._t_measure_minimum = common.build_text_field("0")
        self._c_target_attribute = common.build_combo_box(slot=self._target_attribute_changed)
        self._c_misc_field = common.build_combo_box(slot=self._misc_field_changed)
        self._c_misc_field.setMaxVisibleItems(20)
        self._l_target_attribute = QLabel("<u>p</u>rimary target")
        self._l_target_attribute.setBuddy(self._c_target_attribute)
        self._l_misc_field = QLabel("")
        self._b_multi_regression = common.build_button("Secondary/Tertiary Targets", "Y", self._multi_regression_targets)
        self._l_multi_regression = QLabel("secondary/tertiary targets")
        self._b_multi_targets = common.build_button("Targets and Settings", "T", self._multi_targets_clicked)
        self._l_multi_targets = QLabel("targets and settings")
        self._l_target_info = QLabel("")
        self._l_target_info_text = QLabel("")
        self._l_target_info_text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._l_target_info_text.setWordWrap(True)
        self._b_base_model = common.build_button("Base Model", "M", self._base_model)
        f.addRow("target type", self._c_target_type)
        f.addRow("quality measure", self._c_quality_measure)
        f.addRow("measure minimum", self._t_measure_minimum)
        f.addRow(self._l_target_attribute, self._c_target_attribute)
        f.addRow(self._l_misc_field, self._c_misc_field)
        f.addRow(self._l_multi_regression, self._b_multi_regression)
        f.addRow(self._l_multi_targets, self._b_multi_targets)
        f.addRow(self._l_target_info, self._l_target_info_text)
        f.addRow("", self._b_base_model)
        # TODO MM (java) for stable jar, disable
        self._l_multi_regression.setVisible(False)
        self._b_multi_regression.setVisible(False)
        return box

    def _build_search_conditions_panel(self):
        box = common.build_border("Search Conditions")
        f = self._form(box)
        self._t_depth = common.build_text_field("0")
        self._t_coverage_min = common.build_text_field("0")
        self._t_coverage_max = common.build_text_field("0")
        self._t_minimum_support = common.build_text_field("0")
        self._t_subgroups_max = common.build_text_field("0")
        self._cb_filter = QCheckBox()
        self._cb_filter.setChecked(True)
        self._t_min_improvement = common.build_text_field("0")
        self._t_time_max = common.build_text_field("0")
        f.addRow("refinement depth", self._t_depth)
        f.addRow("minimum coverage", self._t_coverage_min)
        f.addRow("maximum coverage (fraction)", self._t_coverage_max)
        f.addRow("minimum support", self._t_minimum_support)
        f.addRow("maximum subgroups (0 = \u221e)", self._t_subgroups_max)
        f.addRow("filter subgroups", self._cb_filter)
        f.addRow("miminum absolute improvement", self._t_min_improvement)
        f.addRow("maximum time (min) (0 = \u221e)", self._t_time_max)
        return box

    def _build_search_strategy_panel(self):
        box = common.build_border("Search Strategy")
        f = self._form(box)
        self._c_strategy = common.build_combo_box(slot=self._search_strategy_changed)
        self._t_width = common.build_text_field("0")
        self._cb_set_valued = QCheckBox()
        self._c_numeric_strategy = common.build_combo_box(slot=self._numeric_strategy_changed)
        self._c_numeric_operators = common.build_combo_box(slot=self._numeric_operators_changed)
        self._t_nr_bins = common.build_text_field("0")
        self._t_nr_threads = common.build_text_field(str(os.cpu_count() or 1))
        self._t_nr_threads.setToolTip("saved in autorun files; the python port mines on one thread")
        f.addRow("strategy type", self._c_strategy)
        f.addRow("search width", self._t_width)
        f.addRow("set-valued nominals", self._cb_set_valued)
        f.addRow("numeric strategy", self._c_numeric_strategy)
        f.addRow("numeric operators", self._c_numeric_operators)
        f.addRow("number of bins", self._t_nr_bins)
        f.addRow("threads (0 = all available)", self._t_nr_threads)
        return box

    def _init_static_gui_components(self):
        # normal values only, INTERVALS is added when needed (before the
        # target types, as these decide on INTERVALS)
        for n in NumericStrategy.get_normal_values():
            self._c_numeric_strategy.addItem(n.GUI_TEXT, n)
        for t in TargetType:
            if TargetType.is_implemented(t):
                self._c_target_type.addItem(t.GUI_TEXT, t)
        for s in SearchStrategy:
            self._c_strategy.addItem(s.GUI_TEXT, s)
        for n in NumericOperatorSetting.get_normal_values():
            self._c_numeric_operators.addItem(n.DISPLAY_TEXT, n)
        # java: adding the first item fires the listener
        self._search_strategy_changed()
        self._numeric_strategy_changed()
        self._numeric_operators_changed()

    def init_title(self):
        self.setWindowTitle(TITLE)

    # called from the mining thread
    def set_mining_title(self, text):
        self._title_signal.emit(text)

    def _set_title_now(self, text):
        self.setWindowTitle(text)
        self._status.showMessage(text)
        task = getattr(self, "_task", None)
        if task is not None:
            try:
                task.status.setText(text)
            except RuntimeError:
                pass

    def is_cancelled(self):
        return self._is_cancelled

    # ------------------------------------------------------------------
    # filling the fields
    # ------------------------------------------------------------------

    def _init_gui_components(self):
        """only called when a Table is present"""
        self._init_gui_components_data_set(True)
        self._target_type_changed()
        n = self._table.get_nr_rows()
        self._t_depth.setText("1")
        self._t_coverage_min.setText(str(n // 10 + (0 if n % 10 == 0 else 1)))
        self._t_coverage_max.setText("0.9")
        self._t_subgroups_max.setText("1000")
        self._t_time_max.setText("0.0")
        self._t_width.setText("100")
        self._t_nr_bins.setText("8")

    def _init_gui_components_from_file(self):
        sp = self._search_parameters
        self._init_gui_components_data_set(False)
        # search strategy
        self._select(self._c_strategy, sp.get_search_strategy().GUI_TEXT)
        self._t_width.setText(str(sp.get_search_strategy_width()))
        self._cb_set_valued.setChecked(bool(sp.get_nominal_sets()))
        self._select(self._c_numeric_strategy, sp.get_numeric_strategy().GUI_TEXT)
        self._select_data(self._c_numeric_operators, sp.get_numeric_operator_setting())
        self._t_nr_bins.setText(str(sp.get_nr_bins()))
        self._t_nr_threads.setText(str(sp.get_nr_threads()))
        # FIXME MM (java) fields were not disabled correctly after loading, they are now
        self._t_width.setEnabled(sp.get_search_strategy().requires_search_width_parameter())
        self._t_nr_bins.setEnabled(sp.get_numeric_strategy().is_discretiser())
        self._c_numeric_operators.setEnabled(sp.get_numeric_strategy().is_for_half_interval())
        # search conditions
        self._t_depth.setText(str(sp.get_search_depth()))
        self._t_coverage_min.setText(str(sp.get_minimum_coverage()))
        self._t_coverage_max.setText(float_to_string(sp.get_maximum_coverage_fraction()))
        self._t_subgroups_max.setText(str(sp.get_maximum_subgroups()))
        self._t_time_max.setText(float_to_string(sp.get_maximum_time()))
        self._t_minimum_support.setText(str(sp.get_minimum_support()))
        self._cb_filter.setChecked(bool(sp.get_filter_subgroups()))
        self._t_min_improvement.setText(float_to_string(sp.get_minimum_improvement()))

        # target concept, the target type cascade sets up all fields, the
        # loaded values are put back afterwards
        tc = self._target_concept
        tt = tc.get_target_type()
        primary = misc = None
        if TargetType.has_target_attribute(tt):
            primary = tc.get_primary_target().get_name()
            if TargetType.has_target_value(tt):
                if tt == TargetType.SINGLE_NOMINAL:
                    misc = tc.get_target_value()
                elif tt in (TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION,
                            TargetType.DOUBLE_BINARY, TargetType.SCAPE):
                    misc = tc.get_secondary_target().get_name()
        qm_name = sp.get_quality_measure().GUI_TEXT
        qm_min = sp.get_quality_measure_minimum()
        numeric_strategy = sp.get_numeric_strategy()

        self._select(self._c_target_type, tt.GUI_TEXT)
        self._target_type_changed()
        if TargetType.has_target_attribute(tt):
            # NOTE java sets the primary target without refreshing the target
            # values, so a loaded target value of another attribute got lost
            self._select(self._c_target_attribute, primary)
            self._target_attribute_changed()
            if misc is not None:
                self._select(self._c_misc_field, misc)
                self._misc_field_changed()
        self._select(self._c_quality_measure, qm_name)
        sp.set_quality_measure(QM.from_string(qm_name))
        self._t_measure_minimum.setText(float_to_string(qm_min))
        sp.set_quality_measure_minimum(qm_min)
        # 'intervals' only exists in the box after the cascade, java loses
        # it here (the box falls back to its first item)
        if self._c_numeric_strategy.findText(numeric_strategy.GUI_TEXT) >= 0:
            self._select(self._c_numeric_strategy, numeric_strategy.GUI_TEXT)
            self._numeric_strategy_changed()
        # TODO MM (java) some TargetTypes may not be loaded properly

    def _init_gui_components_data_set(self, set_default_target_type):
        if self._table is None:
            return
        t = self._table
        self._l_table_name.setText(t.get_name())
        if self._selection is None:
            self._l_nr_examples.setText(str(t.get_nr_rows()))
        else:
            self._l_nr_examples.setText("%d    (%d in selection)" % (t.get_nr_rows(), int(np.count_nonzero(self._selection))))
        counts = t.get_type_counts()
        totals = [t.get_nr_columns(), sum(c[1] for c in counts)]
        self._l_nr_attributes.setText(_counts_text(totals))
        self._l_nr_nominals.setText(_counts_text(counts[0]))
        self._l_nr_numerics.setText(_counts_text(counts[1]))
        self._l_nr_binaries.setText(_counts_text(counts[3]))
        if not set_default_target_type:
            return
        if counts[0][0] != 0:
            self._select(self._c_target_type, TargetType.SINGLE_NOMINAL.GUI_TEXT)
        elif counts[1][0] != 0:
            self._select(self._c_target_type, TargetType.SINGLE_NUMERIC.GUI_TEXT)
        elif counts[2][0] != 0:
            self._select(self._c_target_type, TargetType.SINGLE_ORDINAL.GUI_TEXT)
        elif counts[3][0] != 0:
            self._select(self._c_target_type, TargetType.MULTI_LABEL.GUI_TEXT)
        else:
            raise AssertionError()

    def _enable_table_dependent_components(self, on):
        for a in (self._m_browse, self._m_explore, self._m_meta, self._m_subset, self._m_sd,
                  self._m_create_autorun, self._m_add_autorun, self._m_sampled, self._m_cui,
                  self._m_go, self._m_custom):
            a.setEnabled(on)
        for b in (self._b_browse, self._b_explore, self._b_meta, self._b_subset, self._b_discretise,
                  self._b_cv, self._b_sd, self._b_sd_loop, self._b_threshold, self._b_multi_targets,
                  self._b_multi_regression):
            b.setEnabled(on)
        self._enable_base_model_button_check()
        for c in (self._c_target_type, self._c_strategy, self._c_numeric_strategy, self._c_numeric_operators):
            c.setEnabled(on)
        self._t_minimum_support.setEnabled(on and self._target_concept.get_target_type() == TargetType.SINGLE_NOMINAL)

    def _init_target_attributes(self):
        tt = self._target_concept.get_target_type()
        self._c_target_attribute.clear()
        self._c_misc_field.clear()
        self._multi_targets.clear()
        self._multi_regression_targets.clear()
        exists_target = exists_class = False
        fill_misc = tt in (TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION,
                           TargetType.DOUBLE_BINARY, TargetType.SCAPE)
        for c in self._table.get_columns():
            at = c.get_type()
            name = c.get_name()
            if name == "target":
                exists_target = True
            if name == "class":
                exists_class = True
            if ((tt == TargetType.SINGLE_NOMINAL and at in (AttributeType.NOMINAL, AttributeType.BINARY)) or
                    (tt == TargetType.SINGLE_NUMERIC and at == AttributeType.NUMERIC) or
                    (tt == TargetType.MULTI_NUMERIC and at == AttributeType.NUMERIC) or
                    (tt == TargetType.SINGLE_ORDINAL and at == AttributeType.NUMERIC) or
                    (tt == TargetType.DOUBLE_REGRESSION and at == AttributeType.NUMERIC) or
                    (tt == TargetType.DOUBLE_CORRELATION and at == AttributeType.NUMERIC) or
                    (tt == TargetType.DOUBLE_BINARY and at == AttributeType.BINARY) or
                    (tt == TargetType.LABEL_RANKING and at == AttributeType.NOMINAL) or
                    (tt == TargetType.MULTI_BINARY_CLASSIFICATION and at in (AttributeType.BINARY, AttributeType.NOMINAL))):
                self._c_target_attribute.addItem(name)
                # not updated when the primary target changes, see init_target_values()
                if fill_misc:
                    self._c_misc_field.addItem(name)
            if tt == TargetType.SCAPE:
                if at == AttributeType.BINARY:
                    self._c_target_attribute.addItem(name)
                elif at == AttributeType.NUMERIC:
                    self._c_misc_field.addItem(name)
        # multi targets do not change with the primary target
        if tt in (TargetType.MULTI_LABEL, TargetType.MULTI_NUMERIC):
            self._set_multi_targets(tt)
        if exists_target:
            self._select(self._c_target_attribute, "target")
        elif exists_class:
            self._select(self._c_target_attribute, "class")

    def _set_multi_regression_targets(self):
        # assumes COOKS_DISTANCE is only valid for DOUBLE_REGRESSION
        if self._quality_measure_name() != QM.COOKS_DISTANCE.GUI_TEXT:
            return
        self._multi_regression_targets.clear()
        primary = self._target_concept.get_primary_target()
        lst = []
        for c in self._table.get_columns():
            if c.get_type() == AttributeType.NUMERIC and c.get_is_enabled() and c is not primary:
                self._multi_regression_targets.add(c.get_name())
                lst.append(c)
        self._target_concept.set_multi_regression_targets(lst)
        self._multi_regression_targets.set_selection_interval(0, len(lst) - 1)

    def _set_multi_targets(self, tt):
        self._multi_targets.clear()
        if tt == TargetType.MULTI_LABEL:
            at = AttributeType.BINARY
        elif tt == TargetType.MULTI_NUMERIC:
            at = AttributeType.NUMERIC
        else:
            raise AssertionError("Unvalid TargetType: " + str(tt))
        lst = []
        for c in self._table.get_columns():
            if c.get_type() == at and c.get_is_enabled():
                self._multi_targets.add(c.get_name())
                lst.append(c)
        self._target_concept.set_multi_targets(lst)
        self._multi_targets.set_selection_interval(0, len(lst) - 1)

    def _init_target_values(self):
        """misc field items for SINGLE_NOMINAL / MULTI_BINARY_CLASSIFICATION"""
        self._c_misc_field.clear()
        if self._c_target_attribute.count() > 0:
            for v in self._table.get_column(self._target_attribute_name()).get_domain():
                self._c_misc_field.addItem(v)

    def _init_target_info(self):
        # in all but one case disabled, SINGLE_NOMINAL enables it again
        self._t_minimum_support.setEnabled(False)
        tt = self._target_concept.get_target_type()
        info, text = self._l_target_info, self._l_target_info_text
        t = self._table
        if tt == TargetType.SINGLE_NOMINAL:
            self._t_minimum_support.setEnabled(True)
            target = self._target_attribute_name()
            misc = self._misc_field_name()
            # may be called before an item is actually selected
            if target is None or misc is None:
                return
            col = t.get_column(target)
            self._positive_count = col.count_values(misc, None)
            if self._selection is not None:
                self._positive_count_selection = col.count_values(misc, self._selection)
            else:
                self._positive_count_selection = self._positive_count
            pct = float(np.float32(np.float32(self._positive_count * 100) / np.float32(t.get_nr_rows())))
            info.setText(" # positive")
            if self._selection is None:
                text.setText("%d (%s%%)" % (self._positive_count, common.number(pct, 2)))
            else:
                n_sel = int(np.count_nonzero(self._selection))
                pct_sel = float(np.float32(np.float32(self._positive_count_selection * 100) / np.float32(n_sel))) if n_sel else float("nan")
                text.setText("%d (%s%%), %d (%s %%) in selection" % (self._positive_count, common.number(pct, 2),
                                                                     self._positive_count_selection, common.number(pct_sel, 2)))
        elif tt == TargetType.SINGLE_NUMERIC:
            target = self._target_attribute_name()
            if target is not None:
                info.setText(" average")
                self._target_average = t.get_column(target).get_average(None)
                if self._selection is None:
                    text.setText(float_to_string(self._target_average))
                else:
                    self._target_average_selection = t.get_column(target).get_average(self._selection)
                    text.setText("%s, %s in selection" % (float_to_string(self._target_average),
                                                          float_to_string(self._target_average_selection)))
        elif tt == TargetType.MULTI_NUMERIC:
            info.setText(" # dimensions")
            text.setText(str(len(self._target_concept.get_multi_targets())))
            # only for 2 targets
            self._enable_base_model_button_check()
        elif tt == TargetType.SINGLE_ORDINAL:
            raise AssertionError(tt)
        elif tt == TargetType.DOUBLE_REGRESSION:
            p = self._target_concept.get_primary_target()
            s = self._target_concept.get_secondary_target()
            if self._quality_measure_name() == QM.COOKS_DISTANCE.GUI_TEXT:
                info.setText(" # multi-targets")
                text.setText(str(self._multi_regression_targets.size()))
            elif p is s:
                # may not be exactly 1.0 (rounding errors)
                info.setText(" regression")
                text.setText("s = 0.0 + 1.0 * p")
            elif p is not None and s is not None:
                rm = RegressionMeasure(QM.REGRESSION_SSD_COMPLEMENT, p, s)
                info.setText(" regression")
                text.setText("s = %s + %s * p" % (common.number(rm.get_intercept(), 2), common.number(rm.get_slope(), 2)))
        elif tt == TargetType.DOUBLE_CORRELATION:
            p = self._target_concept.get_primary_target()
            s = self._target_concept.get_secondary_target()
            if p is not None and s is not None:
                cm = CorrelationMeasure(QM.CORRELATION_R, p, s)
                info.setText(" correlation")
                text.setText(double_to_string(cm.get_evaluation_measure_value()))
        elif tt == TargetType.DOUBLE_BINARY:
            info.setText(" fixme")
            text.setText("value")
        elif tt == TargetType.SCAPE:
            b = self._target_concept.get_primary_target()
            n = self._target_concept.get_secondary_target()
            info.setText(" overall average ranking loss")
            if b is None or n is None:
                text.setText("")
            else:
                self._positive_count = int(np.count_nonzero(b.get_binaries()))
                qm = QualityMeasure(QM.SUBRANKING_LOSS, t.get_nr_rows(), self._positive_count, b, n)
                loss = qm.get_overall_subranking_loss()
                self._search_parameters.set_overall_ranking_loss(loss)
                text.setText(float_to_string(loss))
        elif tt == TargetType.MULTI_LABEL:
            info.setText(" # binary targets")
            text.setText(str(len(self._target_concept.get_multi_targets())))
        elif tt == TargetType.LABEL_RANKING:
            info.setText(" # average ranking")
            col = self._target_concept.get_primary_target()
            if col is not None:
                text.setText(col.get_average_ranking(None).get_ranking())
        elif tt == TargetType.MULTI_BINARY_CLASSIFICATION:
            info.setText(" target info")
            text.setText("none")
        else:
            raise AssertionError(tt)

    def _init_quality_measure(self):
        self._c_quality_measure.clear()
        for q in QM.get_quality_measures(self._target_concept.get_target_type()):
            self._c_quality_measure.addItem(q.GUI_TEXT, q)
        self._init_evaluation_minimum()

    def _init_evaluation_minimum(self):
        name = self._quality_measure_name()
        if name is None:
            return
        if name == QM.AVERAGE.GUI_TEXT:
            self._t_measure_minimum.setText(float_to_string(self._target_average))
        elif name == QM.INVERSE_AVERAGE.GUI_TEXT:
            self._t_measure_minimum.setText(float_to_string(-self._target_average))
        else:
            self._t_measure_minimum.setText(QM.from_string(name).MEASURE_DEFAULT)

    def _init_numeric_strategy(self):
        c = self._c_numeric_strategy
        intervals = NumericStrategy.NUMERIC_INTERVALS.GUI_TEXT
        if self._target_concept.get_target_type() == TargetType.SINGLE_NOMINAL:
            # FIXME (java) horrible check, very likely to break
            if c.count() == len(NumericStrategy.get_normal_values()):
                c.addItem(intervals, NumericStrategy.NUMERIC_INTERVALS)
        else:
            if c.currentText() == intervals:
                c.setCurrentIndex(0)
                self._numeric_strategy_changed()
            i = c.findText(intervals)
            if i >= 0:
                c.removeItem(i)

    def _enable_base_model_button_check(self):
        tt = self._target_concept.get_target_type()
        on = TargetType.has_base_model(tt)
        n = len(self._multi_targets.get_selected_indices())
        if tt == TargetType.MULTI_LABEL and n == 0:
            on = False
        elif tt == TargetType.MULTI_NUMERIC and n != 2:
            on = False
        self._b_base_model.setEnabled(on and self._table is not None)

    def update_table(self):
        """java update(): the Table changed (meta data, enrichment), keep what can be kept

        NOTE java resets the target type here (setTargetTypeName fires the
        cascade), after which only the primary target and the measure minimum
        are put back. Here the target type, primary target, target value /
        secondary target and quality measure are kept when still possible."""
        self._refresh_keeping_choices()

    def _refresh_keeping_choices(self):
        old_type = self._target_type_name()
        primary = self._target_attribute_name()
        misc = self._misc_field_name()
        qm_name = self._quality_measure_name()
        minimum = self._t_measure_minimum.text()
        self._init_gui_components_data_set(False)
        self._select(self._c_target_type, old_type)
        self._target_type_changed()
        if TargetType.has_target_attribute(self._target_concept.get_target_type()) and self._c_target_attribute.count() == 0:
            # nothing left for this target type, fall back to the default
            self._init_gui_components_data_set(True)
            self._target_type_changed()
        if qm_name is not None and self._c_quality_measure.findText(qm_name) >= 0:
            self._select(self._c_quality_measure, qm_name)
        if primary is not None and self._c_target_attribute.findText(primary) >= 0:
            self._select(self._c_target_attribute, primary)
            self._target_attribute_changed()
        if misc is not None and self._c_misc_field.findText(misc) >= 0:
            self._select(self._c_misc_field, misc)
            self._misc_field_changed()
        self._t_measure_minimum.setText(minimum)

    # ------------------------------------------------------------------
    # combo box handlers
    # ------------------------------------------------------------------

    def _target_type_changed(self):
        if self._table is None:
            return
        self._target_concept.set_target_type(self._target_type_name())
        self._search_parameters.set_target_concept(self._target_concept)
        self._init_quality_measure()
        self._init_target_attributes()
        tt = self._target_concept.get_target_type()
        has_misc = TargetType.has_misc_field(tt)
        self._c_misc_field.setVisible(has_misc)
        self._l_misc_field.setVisible(has_misc)
        if tt == TargetType.SINGLE_NOMINAL:
            self._l_misc_field.setText("target value")
        elif tt in (TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION, TargetType.DOUBLE_BINARY):
            self._l_misc_field.setText("<u>s</u>econdary target")
        else:
            self._l_misc_field.setText("")
        # only valid for nominal targets
        self._cb_set_valued.setEnabled(tt == TargetType.SINGLE_NOMINAL)
        has_multi = TargetType.has_multi_targets(tt)
        self._l_multi_targets.setVisible(has_multi)
        self._b_multi_targets.setVisible(has_multi)
        self._b_multi_targets.setEnabled(len(self._multi_targets.get_selected_indices()) != 0)
        self._enable_base_model_button_check()
        has_target = TargetType.has_target_attribute(tt)
        self._l_target_attribute.setVisible(has_target)
        self._c_target_attribute.setVisible(has_target)
        self._init_numeric_strategy()
        self._target_attribute_changed()

    def _quality_measure_changed(self):
        self._init_evaluation_minimum()
        self._search_parameters.set_quality_measure_minimum(self._quality_measure_minimum())
        # this ALWAYS resets alpha when switching TO EDIT_DISTANCE
        if self._quality_measure_name() == QM.EDIT_DISTANCE.GUI_TEXT:
            self._search_parameters.set_alpha(ALPHA_EDIT_DISTANCE)

    def _target_attribute_changed(self):
        name = self._target_attribute_name()
        self._target_concept.set_primary_target(self._table.get_column(name) if name is not None else None)
        self._search_parameters.set_target_concept(self._target_concept)
        tt = self._target_concept.get_target_type()
        if tt not in (TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION, TargetType.DOUBLE_BINARY, TargetType.SCAPE):
            self._init_target_values()
        self._set_multi_regression_targets()
        self._misc_field_changed()

    def _misc_field_changed(self):
        tt = self._target_concept.get_target_type()
        misc = self._misc_field_name()
        if tt in (TargetType.SINGLE_NOMINAL, TargetType.MULTI_BINARY_CLASSIFICATION):
            self._target_concept.set_target_value(misc)
        elif tt in (TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION, TargetType.DOUBLE_BINARY, TargetType.SCAPE):
            self._target_concept.set_secondary_target(self._table.get_column(misc) if misc is not None else None)
        elif tt == TargetType.SINGLE_ORDINAL:
            raise AssertionError(tt)
        self._search_parameters.set_target_concept(self._target_concept)
        self._init_target_info()

    def _search_strategy_changed(self):
        name = self._c_strategy.currentText()
        if name:
            self._search_parameters.set_search_strategy(name)
            self._t_width.setEnabled(self._search_parameters.get_search_strategy().requires_search_width_parameter())

    def _numeric_strategy_changed(self):
        name = self._c_numeric_strategy.currentText()
        if name:
            self._search_parameters.set_numeric_strategy(name)
            s = self._search_parameters.get_numeric_strategy()
            self._t_nr_bins.setEnabled(s.is_discretiser())
            self._c_numeric_operators.setEnabled(s.is_for_half_interval() and self._table is not None)

    def _numeric_operators_changed(self):
        n = self._c_numeric_operators.currentData()
        if n is not None:
            self._search_parameters.set_numeric_operators(n)

    # ------------------------------------------------------------------
    # menu items
    # ------------------------------------------------------------------

    def _open_file(self):
        path = ui_hooks.choose_open_file(FileType.ALL_DATA_FILES)
        if path:
            self.load_file(path)

    def load_file(self, path):
        """loads a data / xml file on a worker thread"""
        def work():
            h = FileHandler(path=path)
            h._set_file(path)
            h._open_file(True)
            return h.get_table(), h.get_search_parameters()

        common.BackgroundTask(self, "Loading...", "Loading '%s'." % os.path.basename(path), work, self._file_loaded).start()

    def _file_loaded(self, res):
        table, sp = res
        if table is None:
            return
        self._multi_targets.clear()
        self._multi_regression_targets.clear()
        self._table = table
        self._selection = None
        if sp is None:   # a regular file
            self._init_gui_components()
        else:            # loaded from xml
            self._search_parameters = sp
            self._target_concept = sp.get_target_concept() if sp.get_target_concept() is not None else TargetConcept()
            self._init_gui_components_from_file()
        self._enable_table_dependent_components(True)
        self._m_remove_domain.setEnabled(False)

    def _load_database_table(self):
        from ..database_connection import DatabaseConnection
        table = None
        try:
            self._dbc = DatabaseConnection("com.mysql.jdbc.Driver", "jdbc:mysql://localhost:3306/psr", "localhost", "psr", "root", "")
            self._dbc.open_connection()
            tables = self._dbc.get_table_view_names()
            name, ok = QInputDialog.getItem(self, "Table selection", "Choose table", tables, 0, False)
            if not ok:
                return
            print("selected table (or view): " + name)
            cur = self._dbc.execute_statement("SELECT * FROM " + name, True, False)
            table = Table.from_result_set(cur, name)
        except Exception as e:
            traceback.print_exc()
            QMessageBox.warning(self, "Load PSR Table", str(e))
        if table is not None:
            self._table = table
            self._multi_targets.clear()
            self._multi_regression_targets.clear()
            self._init_gui_components()
            self._enable_table_dependent_components(True)
            self._m_remove_domain.setEnabled(False)

    def _autorun_file(self, option):
        self._setup_search_parameters()
        XMLAutoRun(self._search_parameters, self._table, option)

    def _load_sampled_subgroups(self):
        QMessageBox.information(self, "Load Sampled Subgroups", "load file from Fraunhofer")
        path = ui_hooks.choose_open_file(FileType.ALL_DATA_FILES)
        if path:
            from ..loader_fraunhofer import LoaderFraunhofer
            LoaderFraunhofer(path, self._table, self._search_parameters)

    def _exit(self):
        log.log_command_line("exit")
        from PySide6.QtWidgets import QApplication
        QApplication.closeAllWindows()
        QApplication.quit()

    def closeEvent(self, e):
        # EXIT_ON_CLOSE
        e.accept()
        from PySide6.QtWidgets import QApplication
        QTimer.singleShot(0, QApplication.closeAllWindows)
        QTimer.singleShot(0, QApplication.quit)

    def _add_enrichment_source(self, enrichment_type):
        FileHandler.add_cui_domain(self._table, enrichment_type)
        self.update_table()
        self._m_remove_domain.setEnabled(len(self._table.get_domain_list()) > 0)

    def _remove_enrichment_source(self):
        domains = SelectionList(self._table.get_domain_list())
        if not show_list_window(RemoveDomainWindow, domains, self):
            return
        # java walks getComponentCount() of the JList here, which is not the
        # number of domains; all selected domains are removed instead
        for i in sorted(domains.get_selected_indices(), reverse=True):
            self._table.remove_domain(i)
        self._table.update()
        self.update_table()
        self._m_remove_domain.setEnabled(len(self._table.get_domain_list()) > 0)

    def _about(self):
        QMessageBox.information(self, "About SubDisc",
                                "SubDisc: Subgroup Discovery Tool r1145\n\n This tool was previously known as Cortana."
                                "\n\nPython port %s (PySide6)." % __version__)

    # ------------------------------------------------------------------
    # data set buttons
    # ------------------------------------------------------------------

    def _browse(self):
        BrowseWindow(self._table, None)

    def _explore(self):
        HistogramWindow(self._table)

    def _meta_data(self):
        MetaDataWindow(self, self._table)

    def _select_subset(self):
        print("========Select Subset=======")
        cond = ConditionWindow(self, self._table.get_columns()).get_result()
        if cond is None:   # closed with 'Delete'
            self._selection = None
            print("selection removed")
        else:
            self._selection = self._table.evaluate(create_list(cond))
            print("size: " + str(int(np.count_nonzero(self._selection))))
        # update selection counts
        self._init_gui_components_data_set(False)
        self._init_target_info()

    def _discretise(self):
        from .discretise import discretise_all
        discretise_all(self._table, self._target_concept)

    def _multi_regression_targets(self):
        self._table.get_column(self._target_attribute_name()).make_primary_target()
        MultiRegressionTargetsWindow(self._search_parameters, self._table, self)

    def _multi_targets_clicked(self):
        tt = self._target_concept.get_target_type()
        if tt == TargetType.MULTI_LABEL:
            show_list_window(MultiTargetsWindow, self._multi_targets, self._search_parameters, self)
        elif tt == TargetType.MULTI_NUMERIC:
            show_list_window(MultiNumericTargetsWindow, self._multi_targets, self)
        else:
            raise AssertionError(tt)
        names = set(self._multi_targets.get_selected_values())
        # in order, the names always occur in the columns
        lst = [c for c in self._table.get_columns() if c.get_name() in names]
        self._target_concept.set_multi_targets(lst)
        self._init_target_info()
        self._enable_base_model_button_check()

    def _base_model(self):
        self._setup_search_parameters()
        tt = self._target_concept.get_target_type()
        tc = self._target_concept
        try:
            if tt == TargetType.SINGLE_NUMERIC:
                target = tc.get_primary_target()
                pdf = self._pdf(target)
                ModelWindow.numeric(target, pdf, None, BASE_MODEL_TEXT, False)
            elif tt == TargetType.MULTI_NUMERIC:
                # TODO MM (java) use ProbabilityDensityFunction_ND
                lst = tc.get_multi_targets()
                if len(lst) != 2:
                    raise AssertionError(str(tt) + " base model only available for 2 dimensions")
                n = self._table.get_nr_rows()
                common.set_busy(True)
                pdf = ProbabilityDensityFunction2D(lst, np.ones(n, dtype=bool))
                common.set_busy(False)
                PDFWindow2D.base(pdf, BASE_MODEL_TEXT, lst[0].get_name(), lst[1].get_name())
            elif tt == TargetType.DOUBLE_REGRESSION:
                p, s = tc.get_primary_target(), tc.get_secondary_target()
                rm = RegressionMeasure(self._search_parameters.get_quality_measure(), p, s)
                ModelWindow.scatter(self._table, p, s, rm, None)   # trend line, no subset
            elif tt == TargetType.DOUBLE_CORRELATION:
                ModelWindow.scatter(self._table, tc.get_primary_target(), tc.get_secondary_target(), None, None)
            elif tt == TargetType.SCAPE:
                b, num = tc.get_primary_target(), tc.get_secondary_target()
                binaries = np.asarray(b.get_binaries(), dtype=bool)
                pdf = self._pdf(num)
                from .result_window import _flip_to_length
                negatives = _flip_to_length(binaries)
                pos = self._sub_pdf(pdf, binaries)
                neg = self._sub_pdf(pdf, negatives)
                ModelWindow.numeric(num, pos, neg, BASE_MODEL_TEXT, True)
            elif tt == TargetType.MULTI_LABEL:
                from ..bayesian import Bayesian
                from ..binary_table import BinaryTable
                lst = tc.get_multi_targets()
                common.set_busy(True)
                bayes = Bayesian(BinaryTable(self._table, lst), lst)
                bayes.climb()
                dag = bayes.get_dag()
                dag.print()
                common.set_busy(False)
                ModelWindow.bayesian_network(dag, 1200, 900)
            elif tt == TargetType.LABEL_RANKING:
                col = tc.get_primary_target()
                lr = col.get_average_ranking(None)   # over the entire data set
                lrm = col.get_average_ranking_matrix(None)
                LabelRankingMatrixWindow(lrm, None, "   " + lr.get_ranking())
            else:
                raise AssertionError(tt)
        except Exception as e:
            common.set_busy(False)
            traceback.print_exc()
            common.show_exception(self, e)

    def _pdf(self, column):
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

    # ------------------------------------------------------------------
    # mining buttons
    # ------------------------------------------------------------------

    def _run_task(self, title, text, fn, on_done=None, stoppable=False):
        def done(res):
            self._task = None
            self.init_title()   # reset the window's title
            if on_done is not None:
                on_done(res)

        def stop():
            self._is_cancelled = True

        self._task = common.BackgroundTask(self, title, text, fn, done, stop if stoppable else None)
        self._task.start()

    def _subgroup_discovery(self):
        self._is_cancelled = False
        self._setup_search_parameters()
        table, selection, sp, n = self._table, self._selection, self._search_parameters, self._nr_threads()
        self._run_task("Mining in progress...",
                       " Press Stop to abort the mining run\n(will show result obtained thus far).",
                       lambda: process.run_subgroup_discovery(table, 0, selection, sp, True, n, self),
                       stoppable=True)

    def _subgroup_discovery_loop(self):
        self._is_cancelled = False
        self._setup_search_parameters()
        target = self._table.get_column(self._target_attribute_name())
        tc, sp, n = self._target_concept, self._search_parameters, self._nr_threads()

        def work():
            for nr, value in enumerate(target.get_domain()):
                log.log_command_line("================= Starting run nr. %d ==================" % nr)
                tc.set_target_value(value)   # temporarily
                sd = process.run_subgroup_discovery(self._table, 0, None, sp, False, n, self)
                if len(sd.get_result()) > 0:
                    xml_save(sd.get_result(), "test_" + value + ".txt", sp.get_target_type())
                    log.log_command_line("File \"test_" + value + ".txt\" saved.")
                else:
                    log.log_command_line("No subgroups found. No file saved.")
                log.log_command_line("================= Finished run nr. %d ==================" % nr)

        self._run_task("Mining in progress...", "Running subgroup discovery for every target value.", work, stoppable=True)

    def _compute_threshold(self):
        tt = self._target_concept.get_target_type()
        setup = RandomQualitiesWindow(tt, self).get_settings()
        if not is_valid_random_qualities_setup(setup):
            return
        self._setup_search_parameters()
        sp, t = self._search_parameters, self._table
        if tt == TargetType.SINGLE_NOMINAL:
            qm = QualityMeasure(sp.get_quality_measure(), t.get_nr_rows(), self._positive_count)
        elif tt == TargetType.SINGLE_NUMERIC:
            target = t.get_column(self._target_attribute_name())
            n = target.size()
            q = sp.get_quality_measure()
            stats = target.get_statistics(None, np.ones(n, dtype=bool), q == QM.MMAD, Stat.COMPL in QM.required_stats(q))
            pdf = self._pdf(target)
            qm = QualityMeasure(q, n, stats.get_subgroup_sum(), stats.get_subgroup_sum_squared_deviations(), pdf)
        elif tt == TargetType.MULTI_NUMERIC:
            QMessageBox.information(self, "Multi Numeric Error", "Not implemented yet!")
            return
        elif tt in (TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION, TargetType.DOUBLE_BINARY):
            qm = None
        elif tt == TargetType.SCAPE:
            qm = QualityMeasure(sp.get_quality_measure(), t.get_nr_rows(), self._positive_count,
                                self._target_concept.get_primary_target(), self._target_concept.get_secondary_target())
        elif tt == TargetType.MULTI_LABEL:
            from ..bayesian import Bayesian
            from ..binary_table import BinaryTable
            bayes = Bayesian(BinaryTable(t, self._target_concept.get_multi_targets()))
            bayes.climb()
            qm = QualityMeasure.multi_label(sp, bayes.get_dag(), t.get_nr_rows())
        elif tt == TargetType.LABEL_RANKING:
            target = t.get_column(self._target_attribute_name())
            lr = target.get_average_ranking(None)
            qm = QualityMeasure(sp.get_quality_measure(), t.get_nr_rows(), lr, LabelRankingMatrix(lr))
        else:
            log.log_command_line("Unable to compute random qualities for " + tt.GUI_TEXT)
            return
        v = Validation(sp, t, self._selection, qm)
        self._run_task("Computing random qualities...", "Computing %s random qualities (%s)." % (setup[1], setup[0]),
                       lambda: v.get_qualities(setup), self._threshold_done)

    def _threshold_done(self, qualities):
        if qualities is None:
            return
        d = NormalDistribution(qualities)
        box = QMessageBox(self)
        box.setWindowTitle("Keep quality measure threshold?")
        box.setIcon(QMessageBox.Question)
        box.setText("The following quality measure thresholds were computed:\n"
                    "1% significance level: " + float_to_string(d.get_one_percent_significance()) + "\n"
                    "5% significance level: " + float_to_string(d.get_five_percent_significance()) + "\n"
                    "10% significance level: " + float_to_string(d.get_ten_percent_significance()) + "\n"
                    "Would you like to keep one of these thresholds as search constraint?")
        buttons = [box.addButton(s, QMessageBox.AcceptRole) for s in
                   ("1% significance", "5% significance", "10% significance", "Ignore statistics")]
        box.setDefaultButton(buttons[0])
        box.exec()
        clicked = box.clickedButton()
        if clicked not in buttons:
            return   # closed, do not print
        k = buttons.index(clicked)
        if k == 0:
            self._t_measure_minimum.setText(float_to_string(d.get_one_percent_significance()))
        elif k == 1:
            self._t_measure_minimum.setText(float_to_string(d.get_five_percent_significance()))
        elif k == 2:
            self._t_measure_minimum.setText(float_to_string(d.get_ten_percent_significance()))
        # may take long, after the window is shown
        for q in sorted(qualities):
            log.log_command_line(double_to_string(d.z_transform(q)))
        log.log_command_line("mu: " + double_to_string(d.get_mu()))
        log.log_command_line("sigma: " + double_to_string(d.get_sigma()))

    def _cross_validate(self):
        r = QMessageBox.question(self, "Store results", "Would you like to store binary tables for each fold in a file?",
                                 QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        if r == QMessageBox.Cancel:
            return
        store = r == QMessageBox.Yes
        import time as _time
        stamp = int(_time.time() * 1000)
        k = 10   # TODO (java) set k from the gui
        table = self._table
        n = self._nr_threads()

        def work():
            cv = CrossValidation(table.get_nr_rows(), k)
            name = table.get_name() + "_folds_" + str(stamp) + ".txt"
            writer = None
            if store:
                try:
                    writer = open(name, "w", encoding="utf-8", newline="")
                except OSError:
                    log.log_command_line("Error on file: " + name)
            try:
                for i in range(k):
                    s = cv.get_set(i, True)
                    t = table.select(s)
                    log.log_command_line("size of fold %d: %d" % (i, t.get_nr_rows()))
                    self._bridge_setup()
                    # java points the one shared target concept at the fold
                    # table and back afterwards, which leaves the fold result
                    # windows with the columns of the full table; every fold
                    # gets its own copy here
                    tc = copy.copy(self._target_concept)
                    tc.update_to_new_table(t)   # point to the temporary table
                    sp = copy.copy(self._search_parameters)
                    sp.set_target_concept(tc)
                    # NOTE java passes the fold selection s (rows of the full table)
                    # together with the fold table t, so the row numbers do not
                    # line up; t holds just the training rows, so no selection
                    sd = process.run_subgroup_discovery(t, i + 1, None, sp, True, n, None)
                    if writer is not None:
                        try:
                            writer.write("Fold %d:\n%s\n" % (i + 1, _bitset_string(s)))
                        except OSError as e:
                            log.log_command_line("File writer error: " + str(e))
                        sd.get_result().save_extent(writer, table, s, tc)
            finally:
                if writer is not None:
                    writer.close()

        self._run_task("Cross-validation...", "Running a 10 fold cross-validation.", work)

    def _bridge_setup(self):
        """setupSearchParameters() reads the widgets, so it has to run on the gui thread"""
        common.bridge().call(self._setup_search_parameters, wait=True)

    def _setup_search_parameters(self):
        tc = self._target_concept
        tt = tc.get_target_type()
        t = self._table
        if TargetType.has_target_attribute(tt):
            tc.set_primary_target(t.get_column(self._target_attribute_name()))
        else:
            tc.set_primary_target(None)
        tc.set_target_value(self._misc_field_name() if tt == TargetType.SINGLE_NOMINAL else None)
        if TargetType.has_secondary_target(tt):
            tc.set_secondary_target(t.get_column(self._misc_field_name()))
        else:
            tc.set_secondary_target(None)
        # set already when needed, remove possible old values
        if not TargetType.has_multi_targets(tt):
            tc.set_multi_targets([])
        # assumes COOKS_DISTANCE is only valid for DOUBLE_REGRESSION
        if self._quality_measure_name() == QM.COOKS_DISTANCE.GUI_TEXT:
            tc.set_multi_regression_targets([])

        sp = self._search_parameters
        sp.set_target_concept(tc)
        sp.set_quality_measure(QM.from_string(self._quality_measure_name()))
        sp.set_quality_measure_minimum(self._quality_measure_minimum())
        sp.set_search_depth(_int(self._t_depth.text(), 1))
        sp.set_minimum_coverage(_int(self._t_coverage_min.text(), 0))
        sp.set_maximum_coverage_fraction(_float(self._t_coverage_max.text(), 1.0))
        sp.set_minimum_support(_int(self._t_minimum_support.text(), 0))
        sp.set_maximum_subgroups(_int(self._t_subgroups_max.text(), 50))
        sp.set_maximum_time(_float(self._t_time_max.text(), 1.0))
        sp.set_search_strategy(self._c_strategy.currentText())
        # last known value, also for BEST_FIRST
        sp.set_search_strategy_width(_int(self._t_width.text(), 100))
        sp.set_nominal_sets(self._cb_set_valued.isChecked())
        sp.set_numeric_strategy(self._c_numeric_strategy.currentText())
        sp.set_numeric_operators(self._c_numeric_operators.currentData())
        # last known value, also for NUMERIC_BINS
        sp.set_nr_bins(_int(self._t_nr_bins.text(), 8))
        sp.set_filter_subgroups(self._cb_filter.isChecked())
        sp.set_minimum_improvement(_float(self._t_min_improvement.text(), 0.0))
        sp.set_nr_threads(self._nr_threads())

    # ------------------------------------------------------------------
    # field access
    # ------------------------------------------------------------------

    @staticmethod
    def _select(combo, text):
        if text is None:
            return
        i = combo.findText(text)
        if i >= 0:
            combo.setCurrentIndex(i)

    @staticmethod
    def _select_data(combo, data):
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _target_type_name(self):
        return self._c_target_type.currentText() or None

    def _quality_measure_name(self):
        return self._c_quality_measure.currentText() or None

    def _quality_measure_minimum(self):
        return _float(self._t_measure_minimum.text(), 0.0)

    def _target_attribute_name(self):
        return self._c_target_attribute.currentText() or None

    def _misc_field_name(self):
        return self._c_misc_field.currentText() or None

    def _nr_threads(self):
        return _int(self._t_nr_threads.text(), os.cpu_count() or 1)


def _counts_text(counts):
    """'12          (10 enabled)', the spacer shrinks with the number of digits"""
    n = counts[0]
    enabled = "" if n == 0 else " (%d enabled)" % counts[1]
    spacer = "          "
    i = 0
    while True:
        n //= 10
        if n <= 0:
            break
        i += 2
    return "%d%s%s" % (counts[0], spacer[i:], enabled)


def _int(text, default):
    try:
        return parse_int(text)
    except Exception:   # TODO MM (java) warning dialog
        return default


def _float(text, default):
    try:
        return parse_float(text)
    except Exception:   # TODO MM (java) warning dialog
        return default


def _bitset_string(mask):
    """BitSet.toString(): {1, 4, 5}"""
    return "{" + ", ".join(str(int(i)) for i in np.flatnonzero(mask)) + "}"
