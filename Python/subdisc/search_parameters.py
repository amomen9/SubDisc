"""All search parameters of one experiment.

FIXME MM (java): none of the setters validate their input.
"""

from . import log
from .java_compat import float_to_string, parse_float, parse_int, parse_boolean, to_f32
from .numeric_operator_setting import NumericOperatorSetting, operators_to_string
from .numeric_strategy import NumericStrategy
from .qm import QM
from .search_strategy import SearchStrategy
from .target_type import TargetType
from .xml_node import add_node_to, text_of

ALPHA_EDIT_DISTANCE = 0.0
ALPHA_DEFAULT = 0.5
BETA_DEFAULT = 1.0
POST_PROCESSING_COUNT_DEFAULT = 20

INT_MIN = -2147483648


def add_line(sb, setting, value):
    sb.append("\t" + setting + " = " + value + "\n")


class SearchParameters:
    def __init__(self):
        # when adding / removing members update add_node_to() and _load_data()
        self._target_concept = None
        self._quality_measure = None
        self._quality_measure_minimum = 0.0
        self._search_depth = 0
        self._minimum_coverage = 0
        self._maximum_coverage_fraction = 0.0
        self._minimum_support = 0
        self._maximum_subgroups = 0
        self._filter_subgroups = False
        self._minimum_improvement = 0.0
        self._maximum_time = 0.0
        self._search_strategy = None
        self._search_strategy_width = 0
        self._nominal_sets = False
        self._numeric_operator_setting = None
        self._numeric_strategy = None
        self._nr_bins = 0
        self._nr_threads = 0
        # no MiningWindow fields for these, but MULTI_LABEL 'Targets and
        # Settings' needs them, so they are not static
        self._alpha = ALPHA_DEFAULT
        self._beta = BETA_DEFAULT
        self._post_processing_do_auto_run = True
        self._post_processing_count = POST_PROCESSING_COUNT_DEFAULT
        self._overall_ranking_loss = 0.0
        # TODO MM (java) add to loadData() and autorun.dtd
        self._beam_seed = None

    @staticmethod
    def from_xml_node(node):
        sp = SearchParameters()
        # the xml constructor in java does not set the defaults above
        sp._alpha = 0.0
        sp._beta = 0.0
        sp._post_processing_do_auto_run = False
        sp._post_processing_count = 0
        sp._load_data(node)
        return sp

    # quality measure
    def get_target_concept(self):
        return self._target_concept

    def set_target_concept(self, tc):
        self._target_concept = tc

    def get_target_type(self):
        return self._target_concept.get_target_type()

    def get_quality_measure(self):
        return self._quality_measure

    def set_quality_measure(self, qm):
        self._quality_measure = qm

    def get_quality_measure_minimum(self):
        return self._quality_measure_minimum

    def set_quality_measure_minimum(self, v):
        self._quality_measure_minimum = to_f32(v)

    # search conditions
    def get_search_depth(self):
        return self._search_depth

    def set_search_depth(self, v):
        self._search_depth = int(v)

    def get_minimum_coverage(self):
        return self._minimum_coverage

    def set_minimum_coverage(self, v):
        if v < 1:
            raise ValueError("theMinimumCoverage must be >= 1")
        self._minimum_coverage = int(v)

    def get_maximum_coverage_fraction(self):
        return self._maximum_coverage_fraction

    def set_maximum_coverage_fraction(self, v):
        self._maximum_coverage_fraction = to_f32(v)

    def get_minimum_support(self):
        return self._minimum_support

    def set_minimum_support(self, v):
        if v < 0:
            raise ValueError("theMinimumSupport must be >= 0")
        self._minimum_support = int(v)

    def get_maximum_subgroups(self):
        return self._maximum_subgroups

    def set_maximum_subgroups(self, v):
        self._maximum_subgroups = int(v)

    def get_filter_subgroups(self):
        return self._filter_subgroups

    def set_filter_subgroups(self, v):
        self._filter_subgroups = bool(v)

    def get_minimum_improvement(self):
        return self._minimum_improvement

    def set_minimum_improvement(self, v):
        self._minimum_improvement = to_f32(v)

    def get_maximum_time(self):
        return self._maximum_time

    def set_maximum_time(self, v):
        self._maximum_time = to_f32(v)

    # search strategy
    def get_search_strategy(self):
        return self._search_strategy

    def set_search_strategy(self, s):
        self._search_strategy = SearchStrategy.from_string(s) if isinstance(s, str) else s

    def get_nominal_sets(self):
        if self._target_concept.get_target_type() != TargetType.SINGLE_NOMINAL:
            return False
        return self._nominal_sets

    def set_nominal_sets(self, v):
        self._nominal_sets = bool(v)

    # FIXME MM (java) numeric strategy should be set to 'in' when applicable
    def get_numeric_operator_setting(self):
        if self._numeric_strategy.is_for_half_interval():
            return self._numeric_operator_setting
        return NumericOperatorSetting.INTERVALS

    def set_numeric_operators(self, v):
        self._numeric_operator_setting = NumericOperatorSetting.from_string(v) if isinstance(v, str) else v

    def get_numeric_strategy(self):
        return self._numeric_strategy

    def set_numeric_strategy(self, v):
        self._numeric_strategy = NumericStrategy.from_string(v) if isinstance(v, str) else v

    def get_search_strategy_width(self):
        return self._search_strategy_width

    def set_search_strategy_width(self, v):
        self._search_strategy_width = int(v)

    def get_nr_bins(self):
        return self._nr_bins

    def set_nr_bins(self, v):
        self._nr_bins = int(v)

    def get_nr_threads(self):
        return self._nr_threads

    def set_nr_threads(self, v):
        self._nr_threads = int(v)

    def get_alpha(self):
        return self._alpha

    def set_alpha(self, v):
        self._alpha = to_f32(v)

    def get_beta(self):
        return self._beta

    def set_beta(self, v):
        self._beta = to_f32(v)

    def get_post_processing_do_auto_run(self):
        return self._post_processing_do_auto_run

    def set_post_processing_do_auto_run(self, v):
        self._post_processing_do_auto_run = bool(v)

    def get_post_processing_count(self):
        return self._post_processing_count

    def set_post_processing_count(self, v):
        self._post_processing_count = int(v)

    def get_beam_seed(self):
        return self._beam_seed

    def set_beam_seed(self, v):
        self._beam_seed = v

    def get_overall_ranking_loss(self):
        return self._overall_ranking_loss

    def set_overall_ranking_loss(self, v):
        self._overall_ranking_loss = to_f32(v)

    def add_node_to(self, parent):
        node = add_node_to(parent, "search_parameters")
        # the TargetConcept writes its own node
        add_node_to(node, "quality_measure", self._quality_measure.GUI_TEXT)
        add_node_to(node, "quality_measure_minimum", float_to_string(self.get_quality_measure_minimum()))
        add_node_to(node, "search_depth", self.get_search_depth())
        add_node_to(node, "minimum_coverage", self.get_minimum_coverage())
        add_node_to(node, "maximum_coverage_fraction", float_to_string(self.get_maximum_coverage_fraction()))
        add_node_to(node, "minimum_support", self.get_minimum_support())
        add_node_to(node, "maximum_subgroups", self.get_maximum_subgroups())
        add_node_to(node, "filter_subgroups", self.get_filter_subgroups())
        add_node_to(node, "minimum_improvement", float_to_string(self.get_minimum_improvement()))
        add_node_to(node, "maximum_time", float_to_string(self.get_maximum_time()))
        add_node_to(node, "search_strategy", self.get_search_strategy().GUI_TEXT)
        add_node_to(node, "use_nominal_sets", self.get_nominal_sets())
        add_node_to(node, "search_strategy_width", self.get_search_strategy_width())
        add_node_to(node, "numeric_operators", self.get_numeric_operator_setting().GUI_TEXT)
        add_node_to(node, "numeric_strategy", self.get_numeric_strategy().GUI_TEXT)
        add_node_to(node, "nr_bins", self.get_nr_bins())
        add_node_to(node, "nr_threads", self.get_nr_threads())
        add_node_to(node, "alpha", float_to_string(self.get_alpha()))
        add_node_to(node, "beta", float_to_string(self.get_beta()))
        add_node_to(node, "post_processing_do_autorun", self.get_post_processing_do_auto_run())
        add_node_to(node, "post_processing_count", self.get_post_processing_count())
        add_node_to(node, "beam_seed", "" if self.get_beam_seed() is None else _beam_seed_string(self.get_beam_seed()))
        add_node_to(node, "overall_ranking_loss", float_to_string(self.get_overall_ranking_loss()))

    def _load_data(self, node):
        for child in node:
            name = child.tag.lower()
            text = text_of(child)
            if name == "quality_measure":
                self._quality_measure = QM.from_string(text)
            elif name == "quality_measure_minimum":
                self._quality_measure_minimum = parse_float(text)
            elif name == "search_depth":
                self._search_depth = parse_int(text)
            elif name == "minimum_coverage":
                self._minimum_coverage = parse_int(text)
            elif name == "maximum_coverage_fraction":
                self._maximum_coverage_fraction = parse_float(text)
            elif name == "minimum_support":
                self._minimum_support = parse_int(text)
            elif name == "maximum_subgroups":
                self._maximum_subgroups = parse_int(text)
            elif name == "filter_subgroups":
                self._filter_subgroups = parse_boolean(text)
            elif name == "minimum_improvement":
                self._minimum_improvement = parse_float(text)
            elif name == "maximum_time":
                self._maximum_time = parse_float(text)
            elif name == "search_strategy":
                self._search_strategy = SearchStrategy.from_string(text)
            elif name == "use_nominal_sets":
                self._nominal_sets = parse_boolean(text)
            elif name == "search_strategy_width":
                self._search_strategy_width = INT_MIN if len(text) == 0 else parse_int(text)
            elif name == "numeric_operators":
                self._numeric_operator_setting = NumericOperatorSetting.from_string(text)
            elif name == "numeric_strategy":
                self._numeric_strategy = NumericStrategy.from_string(text)
            elif name == "nr_bins":
                self._nr_bins = parse_int(text)
            elif name == "nr_threads":
                self._nr_threads = parse_int(text)
            elif name == "alpha":
                self._alpha = parse_float(text)
            elif name == "beta":
                self._beta = parse_float(text)
            elif name == "post_processing_do_autorun":
                self._post_processing_do_auto_run = parse_boolean(text)
            elif name == "post_processing_count":
                self._post_processing_count = parse_int(text)
            elif name == "beam_seed":
                self._beam_seed = None   # FIXME MM (java) see multi-targets
            elif name == "overall_ranking_loss":
                self._overall_ranking_loss = parse_float(text)
            else:
                log.log_command_line("ignoring unknown XML node: " + name)

    def __str__(self):
        sb = ["search_parameters\n"]
        add_line(sb, "quality_measure", self._quality_measure.GUI_TEXT)
        add_line(sb, "quality_measure_minimum", float_to_string(self.get_quality_measure_minimum()))
        add_line(sb, "search_depth", str(self.get_search_depth()))
        add_line(sb, "minimum_coverage", str(self.get_minimum_coverage()))
        add_line(sb, "maximum_coverage_fraction", float_to_string(self.get_maximum_coverage_fraction()))
        add_line(sb, "minimum_support", str(self.get_minimum_support()))
        add_line(sb, "maximum_subgroups", str(self.get_maximum_subgroups()))
        add_line(sb, "minimum_improvement", float_to_string(self.get_minimum_improvement()))
        add_line(sb, "filter_subgroups", "true" if self.get_filter_subgroups() else "false")
        add_line(sb, "maximum_time", float_to_string(self.get_maximum_time()))
        add_line(sb, "search_strategy", self.get_search_strategy().GUI_TEXT)
        add_line(sb, "use_nominal_sets", "true" if self.get_nominal_sets() else "false")
        add_line(sb, "search_strategy_width", str(self.get_search_strategy_width()))
        add_line(sb, "numeric_operators", operators_to_string(self.get_numeric_operator_setting().get_operators()))
        add_line(sb, "numeric_strategy", self.get_numeric_strategy().GUI_TEXT)
        add_line(sb, "nr_bins", str(self.get_nr_bins()))
        add_line(sb, "nr_threads", str(self.get_nr_threads()))
        add_line(sb, "alpha", float_to_string(self.get_alpha()))
        add_line(sb, "beta", float_to_string(self.get_beta()))
        add_line(sb, "post_processing_do_autorun", "true" if self.get_post_processing_do_auto_run() else "false")
        add_line(sb, "post_processing_count", str(self.get_post_processing_count()))
        add_line(sb, "beam_seed", "[]" if self.get_beam_seed() is None else _beam_seed_string(self.get_beam_seed()))
        add_line(sb, "overall_ranking_loss", float_to_string(self.get_overall_ranking_loss()))
        return "".join(sb)


def _beam_seed_string(seed):
    return "[" + ", ".join(str(c) for c in seed) + "]"
