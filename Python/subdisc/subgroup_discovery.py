"""The search: refines candidates level by level and keeps the best subgroups.

The java version runs refinements in a thread pool. Python threads do not
help for this kind of work, so the search runs in one thread, in the order
the java code uses with a single thread; the gui runs the whole search in
a worker thread so the window stays responsive.

Only the code paths java actually uses are here: the deprecated mine()
(theNrThreads < 0) throws for every refinement in java, and the Fltr /
Filter classes are switched off (NEVER_SKIP, itsFilter == null).
"""

import math
import time

import numpy as np

from . import log
from . import condition_list_builder as clb
from .attribute_type import AttributeType
from .candidate import Candidate
from .candidate_queue import CandidateQueue
from .column_condition_bases_builder import (get_column_condition_bases_set, ColumnConditionBasesBinary,
                                             ColumnConditionBasesNominalElementOf, ColumnConditionBasesNominalEquals,
                                             ColumnConditionBasesNumericIntervals, ColumnConditionBasesNumericRegular)
from .condition import Condition
from .condition_base import ConditionBase
from .condition_base_set import ConditionBaseSet
from .interval import Interval
from .java_compat import to_f32, float_compare, div, jsqrt, double_to_string, fdiv
from .numeric_strategy import NumericStrategy
from .operators import Operator
from .qm import QM
from .quality_measure import QualityMeasure, calculate_entropy, calculate_propensity_based
from .search_strategy import SearchStrategy
from .stat_enum import Stat
from .subgroup import Subgroup
from .subgroup_set import SubgroupSet
from .target_type import TargetType
from .tree_set import TreeSet
from .value_set import ValueSet

NO_CANDIDATE_LOG = False        # the log slows mining down a lot
BEST_VALUESET_NO_REFINEMENT_SKIP = True
BEST_INTERVAL_NO_REFINEMENT_SKIP = True
USE_SINGLE_BEST_RESULT_LIKE_BEFORE = False
FOR_REAL_PRINTS = (10, 100)     # CoverRedundancy and JointEntropy for topK

# leave TEMPORARY_CODE at False; when True a PMF is used in single numeric H^2
TEMPORARY_CODE = False
TEMPORARY_CODE_NR_SPLIT_POINTS = -1
TEMPORARY_CODE_USE_EQUAL_WIDTH = False

INTERVAL_NS = 100_000_000       # gui title update throttle
INVALID_NR_TRUE_POSITIVES = -1
LONG_MAX = 9223372036854775807
NEG_INF = float('-inf')
POS_INF = float('inf')

_NUMERIC_BEST = (NumericStrategy.NUMERIC_BEST, NumericStrategy.NUMERIC_BEST_BINS,
                 NumericStrategy.NUMERIC_VIKAMINE_CONSECUTIVE_BEST, NumericStrategy.NUMERIC_INTERVALS)


def _german_number(n):
    """NumberFormat.getNumberInstance(Locale.GERMAN): 1.234.567"""
    return "{:,}".format(int(n)).replace(",", ".")


def _now_ms():
    return int(time.time() * 1000)


class _BestSubgroups:
    """best Subgroup for the candidate set and for the result set (often the same)"""
    __slots__ = ("for_candidate_set", "for_result_set")

    def __init__(self):
        self.for_candidate_set = None
        self.for_result_set = None


def get_next_bin_boundary(total_coverage, current_bin, nr_bins):
    """public in java: MiningWindow uses it too"""
    nb = total_coverage * current_bin
    i = int(nb // nr_bins)
    return i - 1 if (nb % nr_bins) == 0 else i


def is_useful_for_candidate_set(child_coverage, parent_coverage, minimum_coverage, child_depth, maximum_depth):
    return (child_coverage < parent_coverage and child_coverage > minimum_coverage
            and child_coverage > 1 and child_depth < maximum_depth)


def is_useful_for_result_set(child_coverage, parent_coverage, minimum_coverage, maximum_coverage):
    return (child_coverage < parent_coverage and child_coverage >= minimum_coverage
            and child_coverage <= maximum_coverage)


def direct_computation(parent, added_condition, quality_or_qm, child_coverage, nr_true_positives):
    """Subgroup with its scores computed from the counts alone (no members)."""
    if isinstance(quality_or_qm, QualityMeasure):
        score = quality_or_qm.calculate(nr_true_positives, child_coverage)
    else:
        score = quality_or_qm
    # FIXME MM (java) q is only cast to float to make historic results comparable
    q = to_f32(score)
    s = div(float(nr_true_positives), child_coverage)
    t = float(nr_true_positives)
    return parent.get_refined_subgroup_scored(added_condition, q, s, t, child_coverage)


class SubgroupDiscovery:
    # ------------------------------------------------------------------
    # construction, one factory per target type (java: overloaded ctors)
    # ------------------------------------------------------------------

    def _base(self, search_parameters, table, selection, main_window):
        self._search_parameters = search_parameters
        self._table = table
        self._selection = selection
        self._nr_rows = table.get_nr_rows() if selection is None else int(np.count_nonzero(selection))
        self._main_window = main_window
        self._minimum_coverage = search_parameters.get_minimum_coverage()
        self._maximum_coverage = int(self._nr_rows * search_parameters.get_maximum_coverage_fraction())
        self._minimum_support = 0
        self._ignore_quality_minimum = False
        self._quality_measure = None
        self._quality_measure_minimum = search_parameters.get_quality_measure_minimum()
        self._binary_target = None
        self._target_rankings = None
        self._numeric_target = None
        self._primary_column = None
        self._secondary_column = None
        self._base_cm = None
        self._base_rm = None
        self._binary_table = None
        self._targets = None
        self.pdf_nd = None
        self._local_knowledge = None
        self._global_knowledge = None
        self._buffer = None
        self._candidate_count = 0
        self._skip_count = 0
        self._candidate_queue = None
        self._result = None
        self._end_time = -LONG_MAX
        self._then = 0
        self._bound_counts = [0] * 9

    @classmethod
    def single_nominal(cls, search_parameters, table, selection, nr_positive, main_window=None):
        sd = cls.__new__(cls)
        sd._base(search_parameters, table, selection, main_window)
        tc = search_parameters.get_target_concept()
        sd._minimum_support = search_parameters.get_minimum_support()
        sd._quality_measure = QualityMeasure(search_parameters.get_quality_measure(), sd._nr_rows, nr_positive)
        target = tc.get_primary_target()
        cb = ConditionBase(target, Operator.EQUALS)
        value = tc.get_target_value()
        t = target.get_type()
        if t == AttributeType.NOMINAL:
            cond = Condition(cb, value)
        elif t == AttributeType.BINARY:
            if not AttributeType.is_valid_binary_value(value):
                raise ValueError(str(value) + " is not a valid BINARY value")
            cond = Condition(cb, AttributeType.is_valid_binary_true_value(value))
        else:
            raise AssertionError(t)
        all_rows = np.ones(table.get_nr_rows(), dtype=bool)
        sd._binary_target = target.evaluate(all_rows, cond)
        sd._result = SubgroupSet(search_parameters.get_maximum_subgroups(), selection, table.get_nr_rows(), sd._binary_target)
        return sd

    @classmethod
    def single_numeric(cls, search_parameters, table, selection, average, main_window=None):
        from .probability_density_function import ProbabilityDensityFunction2
        sd = cls.__new__(cls)
        sd._base(search_parameters, table, selection, main_window)
        tc = search_parameters.get_target_concept()
        sd._numeric_target = tc.get_primary_target()
        stats = sd._numeric_target.get_statistics(selection, None, False,
                                                  Stat.COMPL in QM.required_stats(search_parameters.get_quality_measure()))
        stats.print()
        pdf = ProbabilityDensityFunction2(sd._numeric_target, selection)
        pdf.smooth()
        sd._quality_measure = QualityMeasure(search_parameters.get_quality_measure(), stats.get_coverage(),
                                             stats.get_subgroup_sum(), stats.get_subgroup_sum_squared_deviations(), pdf)
        sd._result = SubgroupSet(search_parameters.get_maximum_subgroups(), selection, table.get_nr_rows())
        return sd

    @classmethod
    def double_target(cls, search_parameters, table, selection, is_regression, main_window=None):
        """DOUBLE_CORRELATION, DOUBLE_REGRESSION and DOUBLE_BINARY"""
        sd = cls.__new__(cls)
        sd._base(search_parameters, table, selection, main_window)
        # ensure theCountHeadBody <= theCoverage
        sd._quality_measure = QualityMeasure(search_parameters.get_quality_measure(), sd._nr_rows, sd._nr_rows)
        if search_parameters.get_quality_measure() == QM.COOKS_DISTANCE:
            log.REFINEMENTLOG = True
            log.open_file_output_streams()
        tc = search_parameters.get_target_concept()
        sd._primary_column = tc.get_primary_target()
        sd._secondary_column = tc.get_secondary_target()
        if search_parameters.get_target_type() != TargetType.DOUBLE_BINARY:
            if is_regression:
                from .regression_measure import RegressionMeasure
                sd._base_rm = RegressionMeasure(search_parameters.get_quality_measure(), sd._primary_column, sd._secondary_column)
                sd._buffer = TreeSet()
            else:
                from .correlation_measure import CorrelationMeasure
                sd._base_cm = CorrelationMeasure(search_parameters.get_quality_measure(), sd._primary_column, sd._secondary_column)
        sd._result = SubgroupSet(search_parameters.get_maximum_subgroups(), selection, table.get_nr_rows())
        return sd

    @classmethod
    def scape(cls, main_window, search_parameters, table, selection):
        sd = cls.__new__(cls)
        sd._base(search_parameters, table, selection, main_window)
        tc = search_parameters.get_target_concept()
        sd._primary_column = tc.get_primary_target()
        sd._secondary_column = tc.get_secondary_target()
        sd._quality_measure = QualityMeasure(search_parameters.get_quality_measure(), sd._nr_rows,
                                             int(np.count_nonzero(sd._primary_column.get_binaries())),
                                             sd._primary_column, sd._secondary_column)
        sd._result = SubgroupSet(search_parameters.get_maximum_subgroups(), selection, table.get_nr_rows())
        return sd

    @classmethod
    def multi_label(cls, search_parameters, table, selection, main_window=None):
        from .binary_table import BinaryTable
        from .bayesian import Bayesian
        sd = cls.__new__(cls)
        sd._base(search_parameters, table, selection, main_window)
        sd._targets = search_parameters.get_target_concept().get_multi_targets()
        sd._binary_table = BinaryTable(table, sd._targets)
        bayes = Bayesian(sd._binary_table, sd._targets)
        bayes.climb()
        sd._quality_measure = QualityMeasure.multi_label(search_parameters, bayes.get_dag(), sd._nr_rows)
        sd._result = SubgroupSet(search_parameters.get_maximum_subgroups(), selection, table.get_nr_rows())
        return sd

    @classmethod
    def label_ranking(cls, search_parameters, main_window, table, selection):
        sd = cls.__new__(cls)
        sd._base(search_parameters, table, selection, main_window)
        tc = search_parameters.get_target_concept()
        sd._target_rankings = tc.get_primary_target()
        lr = sd._target_rankings.get_average_ranking(None)
        lrm = sd._target_rankings.get_average_ranking_matrix(None)
        sd._quality_measure = QualityMeasure(search_parameters.get_quality_measure(), sd._nr_rows, lr, lrm)
        sd._result = SubgroupSet(search_parameters.get_maximum_subgroups(), selection, table.get_nr_rows())
        return sd

    @classmethod
    def multi_numeric(cls, table, selection, main_window, search_parameters):
        from .probability_density_function_nd import ProbabilityDensityFunctionND
        sd = cls.__new__(cls)
        sd._base(search_parameters, table, selection, main_window)
        sd._targets = search_parameters.get_target_concept().get_multi_targets()
        sd.pdf_nd = ProbabilityDensityFunctionND(list(sd._targets))
        sd._quality_measure = None
        sd._result = SubgroupSet(search_parameters.get_maximum_subgroups(), selection, table.get_nr_rows())
        return sd

    def ignore_quality_minimum(self):
        """swap-randomisation and random qualities do not use the minimum"""
        self._ignore_quality_minimum = True

    # ------------------------------------------------------------------
    # mining
    # ------------------------------------------------------------------

    def mine(self, begin_time, nr_threads=1):
        """begin_time in ms (time.time() * 1000)"""
        self._pre_mining(begin_time, nr_threads)
        ccbs = get_column_condition_bases_set(self._table, self._search_parameters)
        if nr_threads < 0:
            # java runs its deprecated mine() here, which throws an
            # AssertionError on the first refinement; use the normal loop
            log.log_command_line("SubgroupDiscovery.mine(): nrThreads < 0 is not supported, running the normal search")
        queue = self._candidate_queue
        while not self._is_time_to_stop():
            if queue.current_level_queue_size() > 0:
                candidate = queue.remove_first()
            elif queue.size() > 0:
                candidate = queue.remove_first()
            else:
                break
            if candidate is None:
                break
            self._test(candidate.get_subgroup(), ccbs)
        self._post_mining(begin_time)
        for top_k in FOR_REAL_PRINTS:
            self._result.post_process_get_cover_redundancy_and_joint_entropy(top_k)

    def _pre_mining(self, begin_time, nr_threads):
        conditions = ConditionBaseSet(self._table, self._search_parameters)
        self._log_experiment_settings(conditions)
        self._load_external_knowledge(conditions)

        start = Subgroup(clb.empty_list(), self._result.get_all_data_bit_set_clone(), self._result)
        if self._is_direct_single_binary():
            start.set_tertiary_statistic(self._quality_measure.get_nr_positives())
        if self._search_parameters.get_beam_seed() is None or nr_threads < 0:
            self._candidate_queue = CandidateQueue(self._search_parameters, Candidate(start))
        else:
            self._candidate_queue = self._get_candidate_queue_from_beam_seed()

        _prepare_data(self._binary_target if self._is_direct_single_binary() else None, self._table.get_columns())

        end = begin_time + int(float(self._search_parameters.get_maximum_time()) * 60.0 * 1000.0)
        self._end_time = LONG_MAX if end <= begin_time else end
        return conditions

    def _load_external_knowledge(self, condition_base_set):
        qm = self._search_parameters.get_quality_measure()
        if qm in (QM.PROP_SCORE_WRACC, QM.PROP_SCORE_RATIO):
            from .knowledge import ExternalKnowledgeFileLoader, LocalKnowledge, GlobalKnowledge
            e = ExternalKnowledgeFileLoader(self._table, condition_base_set)
            self._local_knowledge = LocalKnowledge(self._binary_target, e.get_knowledge(False))
            self._global_knowledge = GlobalKnowledge(self._binary_target, e.get_knowledge(True))
            log.log_command_line("")

    def _log_experiment_settings(self, condition_base_set):
        log.log_command_line("")
        log.log_command_line(str(self._search_parameters.get_target_concept()))
        log.log_command_line(str(self._search_parameters))
        log.log_command_line(str(condition_base_set))

    def _is_direct_single_binary(self):
        """direct computation from true positive counts, SINGLE_NOMINAL only
        (not for the propensity score measures and the vm-* strategies)"""
        s = self._search_parameters
        return (s.get_target_type() == TargetType.SINGLE_NOMINAL
                and s.get_quality_measure() not in (QM.PROP_SCORE_WRACC, QM.PROP_SCORE_RATIO)
                and s.get_numeric_strategy() not in (NumericStrategy.NUMERIC_VIKAMINE_CONSECUTIVE_ALL,
                                                     NumericStrategy.NUMERIC_VIKAMINE_CONSECUTIVE_BEST))

    def _get_candidate_queue_from_beam_seed(self):
        seed = self._search_parameters.get_beam_seed()
        first_cl = seed[0]
        # TODO (java) there may be no members, then this crashes
        first_members = self._table.evaluate(first_cl)
        first = Subgroup(first_cl, first_members, self._result)
        queue = CandidateQueue(self._search_parameters, Candidate(first))
        if first.get_coverage() <= 1:
            queue.remove_first()
        empty = 0
        for cl in seed:
            log.log_command_line(str(cl))
            members = self._table.evaluate(cl)
            if members.any():
                queue.add(Candidate(Subgroup(cl, members, self._result)))
            else:
                empty += 1
        if empty > 0:
            log.log_command_line("Number of empty seeds discarded: " + str(empty))
        log.log_command_line("Beam Seed size: " + str(queue.size()))
        return queue

    def _is_time_to_stop(self):
        if _now_ms() > self._end_time:
            return True
        mw = self._main_window
        if mw is not None and mw.is_cancelled():
            return True
        return False

    def _set_title(self, subgroup):
        mw = self._main_window
        if mw is None:
            return
        now = time.perf_counter_ns()
        if now - self._then < INTERVAL_NS:
            return
        current = str(subgroup)
        mw.set_mining_title("d=" + str(subgroup.get_depth()) + " cands=" + _german_number(self._candidate_count)
                            + " evaluating: " + current)
        self._then = now

    # ------------------------------------------------------------------
    # one candidate (java: Test.run())
    # ------------------------------------------------------------------

    def _test(self, subgroup, ccbs):
        parent_members = subgroup.get_members()
        for ccb in ccbs:
            if self._is_time_to_stop():
                break
            if isinstance(ccb, ColumnConditionBasesBinary):
                self._evaluate_binary(subgroup, parent_members, ccb)
            elif isinstance(ccb, ColumnConditionBasesNominalEquals):
                self._evaluate_nominal_equals(subgroup, parent_members, ccb)
            elif isinstance(ccb, ColumnConditionBasesNominalElementOf):
                self._evaluate_nominal_element_of(subgroup, parent_members, ccb)
            elif isinstance(ccb, ColumnConditionBasesNumericRegular):
                self._evaluate_numeric_regular(subgroup, parent_members, ccb)
            elif isinstance(ccb, ColumnConditionBasesNumericIntervals):
                self._evaluate_numeric_intervals(subgroup, parent_members, ccb)
            else:
                raise AssertionError("Test.run() unexpected subclass of ColumnConditionBases")

    # ------------------------------------------------------------------
    # when done
    # ------------------------------------------------------------------

    def _post_mining(self, begin_time):
        if self._search_parameters.get_filter_subgroups():
            self._result.filter_subgroups(self._search_parameters.get_minimum_improvement())
        nr_subgroups = self.get_number_of_subgroups()
        if self._search_parameters.get_search_strategy() == SearchStrategy.COVER_BASED_BEAM_SELECTION:
            self._post_process_cbss()
        elapsed = _now_ms() - begin_time
        from .process import echo_mining_end
        echo_mining_end(elapsed, nr_subgroups)
        _finish_title(self._main_window, elapsed, self._candidate_count)
        for c in self._table.get_columns():
            c.remove_sorted()
        log.log_command_line("number of candidates: " + str(self._candidate_count))
        self._post_process_cook()
        log.log_command_line("number of subgroups : " + str(self.get_number_of_subgroups()))
        # 1 to n, for future reference in subsets
        self._result.set_ids()
        self._post_process_multi_label_auto_run()

    def _post_process_cook(self):
        if self._search_parameters.get_quality_measure() != QM.COOKS_DISTANCE:
            return
        names = ("seven", "six  ", "five ", "four ")
        for k, nm in enumerate(names):
            log.log_command_line("Bound " + nm + " computed " + str(self._bound_counts[k]) + " times")
        for k, nm in enumerate(names):
            log.log_command_line("Bound " + nm + " fired " + str(self._bound_counts[4 + k]) + " times")
        log.log_command_line("Rank deficient models: " + str(self._bound_counts[8]))

    def _flush_buffer(self):
        if self._buffer is None:
            return
        for c in self._buffer:
            self._check_and_log(c.get_subgroup(), self._nr_rows)
        self._buffer = TreeSet()

    def _post_process_cbss(self):
        s = self._result.post_process(self._search_parameters.get_search_strategy())
        # FIXME MM (java) hack around the odd postProcess implementation
        if self._result is not s:
            self._result.clear()
            self._result.add_all(s)

    def _post_process_multi_label_auto_run(self):
        sp = self._search_parameters
        if not (sp.get_target_type() == TargetType.MULTI_LABEL and sp.get_post_processing_do_auto_run()):
            return
        if self._result.is_empty():
            return
        from .bayesian import Bayesian
        log.log_command_line("Creating quality measures.")
        count = sp.get_post_processing_count()
        count_sq = math.pow(count, 2.0)
        qms = []
        for _ in range(count):
            gb = Bayesian(self._binary_table)
            gb.climb()
            qms.append(QualityMeasure.multi_label(sp, gb.get_dag(), self._nr_rows))
        new_set = SubgroupSet(sp.get_maximum_subgroups(), self._selection, self._nr_rows)
        for s in self._result:
            log.log_command_line("Postprocessing subgroup " + str(s.get_id()))
            total = 0.0
            sub_table = self._binary_table.select_rows(s.get_members())
            for _ in range(count):
                lb = Bayesian(sub_table)
                lb.climb()
                s.set_dag(lb.get_dag())
                for j in range(count):
                    total += qms[j].calculate(s)
            s.set_measure_value(div(total, count_sq))
            s.renounce_p_value()
            new_set.add(s)
        new_set.set_ids()
        self._result.clear()
        self._result.add_all(new_set)

    # ------------------------------------------------------------------
    # binary
    # ------------------------------------------------------------------

    def _evaluate_binary(self, parent, parent_members, ccb):
        parent_cov = parent.get_coverage()
        cb = ccb.get(0)
        col = cb.get_column()
        child = col.evaluate_binary(parent_members, False)
        child_cov = int(np.count_nonzero(child))
        # ignore both f and t
        if child_cov == 0 or child_cov == parent_cov:
            return
        tps = INVALID_NR_TRUE_POSITIVES
        # (col = false)
        if child_cov >= self._minimum_coverage:
            tps = self._evaluate_binary_helper(parent, Condition(cb, False), child, child_cov)
        if self._is_time_to_stop():
            return
        # (col = true), everything that is not false
        child_cov = parent_cov - child_cov
        if child_cov >= self._minimum_coverage:
            cond = Condition(cb, True)
            if tps != INVALID_NR_TRUE_POSITIVES:
                tps = int(parent.get_tertiary_statistic()) - tps
                c = direct_computation(parent, cond, self._quality_measure, child_cov, tps)
                self._check_and_log(c, parent_cov)
            else:
                child = col.evaluate_binary(parent_members, True)
                self._evaluate_binary_helper(parent, cond, child, child_cov)

    def _evaluate_binary_helper(self, parent, added, child_members, child_coverage):
        """INVALID_NR_TRUE_POSITIVES unless in the direct setting"""
        tps = INVALID_NR_TRUE_POSITIVES
        if self._is_direct_single_binary():
            tps = int(np.count_nonzero(child_members & self._binary_target))
            child = direct_computation(parent, added, self._quality_measure, child_coverage, tps)
        else:
            child = parent.get_refined_subgroup(added, child_members, child_coverage)
        self._check_and_log(child, parent.get_coverage())
        return tps

    # ------------------------------------------------------------------
    # nominal
    # ------------------------------------------------------------------

    def _evaluate_nominal_equals(self, parent, parent_members, ccb):
        parent_cov = parent.get_coverage()
        cb = ccb.get(0)
        col = cb.get_column()
        counts = col.get_unique_nominal_domain_counts(parent_members, parent_cov)
        nr_distinct = int(counts[-1])
        # no useful refinement possible, it would have the parent coverage
        if nr_distinct <= 1:
            return
        domain = col.get_shared_domain()
        i = 0
        j = nr_distinct
        while j > 0 and not self._is_time_to_stop():
            count = int(counts[i])
            if count == 0:
                i += 1
                continue
            j -= 1
            if count < self._minimum_coverage:
                i += 1
                continue
            if count == parent_cov:
                break
            cond = Condition(cb, domain[i])
            child = parent.get_refined_subgroup(cond)
            self._check_and_log(child, parent_cov)
            i += 1

    def _evaluate_nominal_element_of(self, parent, parent_members, ccb):
        """BestValueSet, SINGLE_NOMINAL only"""
        from .nominal_cross_table import NominalCrossTable
        parent_cov = parent.get_coverage()
        cb = ccb.get(0)
        nct = NominalCrossTable(cb.get_column(), parent_members, self._binary_target)
        best_subset = set()
        qm = self._search_parameters.get_quality_measure()
        if qm in (QM.WRACC, QM.CORTANA_QUALITY):
            # integer arithmetic instead of Michael's floating point version
            T = self._quality_measure.get_nr_positives()
            N = self._quality_measure.get_nr_records()
            a_p = 0
            a_n = 0
            for i in range(nct.size()):
                if self._is_time_to_stop():
                    break
                pi = nct.get_positive_count(i)
                ni = nct.get_negative_count(i)
                cnt = pi + ni
                # include WRAcc=0 values too: same WRAcc, higher support
                if pi * N >= T * cnt:
                    a_p += pi
                    a_n += ni
                    best_subset.add(nct.get_value(i))
            if not best_subset:
                return
            if (a_p + a_n) == parent_cov:
                self._skip_count += 1
                if BEST_VALUESET_NO_REFINEMENT_SKIP:
                    return
            count_head_body = a_p
            child_cov = a_p + a_n
            final_quality = self._quality_measure.calculate(count_head_body, child_cov)
        else:
            best_q = NEG_INF
            best_p = -2147483648
            best_n = -2147483648
            sorted_idx = nct.get_sorted_domain_indices()
            size = len(sorted_idx)
            # upper part of the hull
            a_p = 0
            a_n = 0
            prev_best = -1
            for i in range(size - 1):
                if self._is_time_to_stop():
                    break
                idx = sorted_idx[i]
                pi = nct.get_positive_count(idx)
                ni = nct.get_negative_count(idx)
                a_p += pi
                a_n += ni
                nxt = sorted_idx[i + 1]
                # skip degenerate hull points
                if i < size - 2 and _i32(pi * nct.get_negative_count(nxt)) == _i32(nct.get_positive_count(nxt) * ni):
                    continue
                q = self._quality_measure.calculate(a_p, a_p + a_n)
                if q > best_q:
                    best_q = q
                    best_p = a_p
                    best_n = a_n
                    for j in range(prev_best + 1, i + 1):
                        best_subset.add(nct.get_value(sorted_idx[j]))
                    prev_best = i
            # lower part of the hull
            a_p = 0
            a_n = 0
            prev_best = -1
            i = size - 1
            while i > 0 and not self._is_time_to_stop():
                idx = sorted_idx[i]
                pi = nct.get_positive_count(idx)
                ni = nct.get_negative_count(idx)
                a_p += pi
                a_n += ni
                prv = sorted_idx[i - 1]
                if i > 1 and _i32(pi * nct.get_negative_count(prv)) == _i32(nct.get_positive_count(prv) * ni):
                    i -= 1
                    continue
                q = self._quality_measure.calculate(a_p, a_p + a_n)
                if q > best_q:
                    best_q = q
                    best_p = a_p
                    best_n = a_n
                    if prev_best == -1:
                        best_subset.clear()
                        prev_best = size
                    for j in range(prev_best - 1, i - 1, -1):
                        best_subset.add(nct.get_value(sorted_idx[j]))
                    prev_best = i
                i -= 1
            if not best_subset:
                return
            # sic: (aP + aN) here is the sum of the last loop, as in java
            if (a_p + a_n) == parent_cov:
                self._skip_count += 1
                if BEST_VALUESET_NO_REFINEMENT_SKIP:
                    return
            count_head_body = best_p
            child_cov = best_p + best_n
            final_quality = best_q
        # FIXME MM (java) this does not take maximum coverage into account
        cond = Condition(cb, ValueSet(best_subset))
        child = direct_computation(parent, cond, final_quality, child_cov, count_head_body)
        self._check_and_log(child, parent_cov)

    # ------------------------------------------------------------------
    # numeric
    # ------------------------------------------------------------------

    def _evaluate_numeric_regular(self, parent, parent_members, ccb):
        e = ccb.get(0)
        l = ccb.get(1)
        g = ccb.get(2)
        col = (e if e is not None else (l if l is not None else g)).get_column()
        if self._is_direct_single_binary():
            v = col.get_unique_numeric_domain_map_tp(parent_members)
            if e is not None:
                self._evaluate_numeric_regular_single_binary(parent, e, v)
            if l is not None:
                self._evaluate_numeric_regular_single_binary(parent, l, v)
            if g is not None:
                self._evaluate_numeric_regular_single_binary(parent, g, v)
        else:
            v = col.get_value_count(parent_members)
            if e is not None:
                self._evaluate_numeric_regular_generic(parent, e, v)
            if l is not None:
                self._evaluate_numeric_regular_generic(parent, l, v)
            if g is not None:
                self._evaluate_numeric_regular_generic(parent, g, v)

    def _evaluate_numeric_regular_generic(self, parent, cb, value_info):
        ns = self._search_parameters.get_numeric_strategy()
        if ns.is_discretiser():
            self._evaluate_numeric_regular_generic_coarse(parent, cb, value_info)
            return
        parent_cov = parent.get_coverage()
        col = cb.get_column()
        op = cb.get_operator()
        is_all = ns == NumericStrategy.NUMERIC_ALL
        best = None if is_all else _BestSubgroups()
        counts = value_info.itsCounts
        n = len(counts)
        sorted_vals = col.sorted_floats_view()
        if op == Operator.EQUALS:
            for i in range(n):
                if self._is_time_to_stop():
                    break
                cnt = int(counts[i])
                if cnt < self._minimum_coverage:
                    continue
                if cnt == parent_cov:
                    break
                cond = Condition(cb, float(sorted_vals[i]), i)
                self._evaluate_candidate_generic(parent, cond, cnt, is_all, best)
        elif op == Operator.LESS_THAN_OR_EQUAL:
            cover = 0
            for i in range(n):
                if self._is_time_to_stop():
                    break
                cnt = int(counts[i])
                if cnt == 0:
                    continue
                cover += cnt
                if cover == parent_cov:
                    break
                if cover < self._minimum_coverage:
                    continue
                cond = Condition(cb, float(sorted_vals[i]), i)
                self._evaluate_candidate_generic(parent, cond, cover, is_all, best)
        elif op == Operator.GREATER_THAN_OR_EQUAL:
            cover = parent_cov
            for i in range(n):
                if self._is_time_to_stop():
                    break
                if cover < self._minimum_coverage:
                    break
                cnt = int(counts[i])
                if cnt == 0:
                    continue
                # >= with the first value selects the same as the parent
                if cover != parent_cov and not math.isnan(float(sorted_vals[i])):
                    cond = Condition(cb, float(sorted_vals[i]), i)
                    self._evaluate_candidate_generic(parent, cond, cover, is_all, best)
                cover -= cnt
        else:
            raise AssertionError("SubgroupDiscovery.evaluateNumericRegularGeneric() + " + str(op))
        if not is_all:
            self._check_and_log_best(best, parent_cov)

    def _evaluate_numeric_regular_generic_coarse(self, parent, cb, value_info):
        ns = self._search_parameters.get_numeric_strategy()
        op = cb.get_operator()
        nr_bins = self._search_parameters.get_nr_bins()
        # not checked in MiningWindow/XML, do nothing (no error)
        if nr_bins <= 1:
            return
        parent_cov = parent.get_coverage()
        b = 1
        col = cb.get_column()
        counts = value_info.itsCounts
        n = len(counts)
        sv = col.sorted_floats_view()
        is_all = ns == NumericStrategy.NUMERIC_BINS
        best = None if is_all else _BestSubgroups()
        if op == Operator.BETWEEN:
            last_cover = 0
            f = NEG_INF
            nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
            cover = 0
            i = 0
            while i < n and b < nr_bins and not self._is_time_to_stop():
                cnt = int(counts[i])
                if cnt == 0:
                    i += 1
                    continue
                cover += cnt
                if cover <= nxt:
                    i += 1
                    continue
                if cover == parent_cov:
                    break
                nv = float(sv[i])
                cond = Condition(cb, Interval(f, nv))
                self._evaluate_candidate_generic(parent, cond, cover - last_cover, is_all, best)
                last_cover = cover
                f = nv
                b += 1
                nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
                while nxt <= cover - 1:
                    b += 1
                    nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
                i += 1
            if last_cover != parent_cov and float_compare(POS_INF, f) != 0:
                cond = Condition(cb, Interval(f, POS_INF))
                self._evaluate_candidate_generic(parent, cond, parent_cov - last_cover, is_all, best)
        elif op == Operator.LESS_THAN_OR_EQUAL:
            nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
            cover = 0
            i = 0
            while b < nr_bins and not self._is_time_to_stop():
                cnt = int(counts[i])
                if cnt == 0:
                    i += 1
                    continue
                cover += cnt
                if cover <= nxt:
                    i += 1
                    continue
                if cover == parent_cov:
                    break
                cond = Condition(cb, float(sv[i]), i)
                self._evaluate_candidate_generic(parent, cond, cover, is_all, best)
                b += 1
                nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
                while nxt <= cover - 1:
                    b += 1
                    nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
                i += 1
        elif op == Operator.GREATER_THAN_OR_EQUAL:
            # NOTE division in old code rounds down for <= and >=
            nxt = int(parent_cov - (parent_cov // nr_bins))
            cover = parent_cov
            i = 0
            while b < nr_bins and not self._is_time_to_stop():
                if cover < self._minimum_coverage:
                    break
                cnt = int(counts[i])
                if cnt == 0:
                    i += 1
                    continue
                if cover - cnt < nxt and cover != parent_cov and not math.isnan(float(sv[i])):
                    cond = Condition(cb, float(sv[i]), i)
                    self._evaluate_candidate_generic(parent, cond, cover, is_all, best)
                    b += 1
                    nxt = int(parent_cov - ((b * parent_cov) // nr_bins))
                    while nxt > (cover - cnt):
                        b += 1
                        nxt = int(parent_cov - ((b * parent_cov) // nr_bins))
                cover -= cnt
                i += 1
        else:
            raise AssertionError("SubgroupDiscovery.evaluateNumericRegularGenericCoarse() + " + str(op))
        if not is_all:
            self._check_and_log_best(best, parent_cov)

    def _evaluate_numeric_regular_single_binary(self, parent, cb, value_info):
        ns = self._search_parameters.get_numeric_strategy()
        if ns.is_discretiser():
            self._evaluate_numeric_regular_single_binary_coarse(parent, cb, value_info)
            return
        parent_cov = parent.get_coverage()
        col = cb.get_column()
        op = cb.get_operator()
        is_all = ns == NumericStrategy.NUMERIC_ALL
        best = None if is_all else _BestSubgroups()
        counts = value_info.itsCounts.tolist()
        tps = value_info.itsTruePositives.tolist()
        sv = col.sorted_floats_view().tolist()
        size = len(counts)
        min_cov = self._minimum_coverage
        min_sup = self._minimum_support
        if op == Operator.EQUALS:
            for i in range(size):
                if self._is_time_to_stop():
                    break
                cnt = counts[i]
                if cnt < min_cov:
                    continue
                if cnt == parent_cov:
                    break
                if not math.isnan(sv[i]):
                    cond = Condition(cb, sv[i], i)
                    self._evaluate_candidate_direct(parent, cond, cnt, tps[i], is_all, best)
        elif op == Operator.LESS_THAN_OR_EQUAL:
            cover = 0
            tp = 0
            for i in range(size):
                if self._is_time_to_stop():
                    break
                cnt = counts[i]
                if cnt == 0:
                    continue
                cover += cnt
                tp += tps[i]
                if cover == parent_cov:
                    break
                if cover < min_cov or tp < min_sup:
                    continue
                if not math.isnan(sv[i]):
                    cond = Condition(cb, sv[i], i)
                    self._evaluate_candidate_direct(parent, cond, cover, tp, is_all, best)
        elif op == Operator.GREATER_THAN_OR_EQUAL:
            cover = parent_cov - value_info.itsMissingCount
            # NOTE getTertiaryStatistic() only works for SINGLE_NOMINAL
            tp = int(parent.get_tertiary_statistic()) - value_info.itsMissingPositiveCount
            for i in range(size):
                if self._is_time_to_stop():
                    break
                if cover < min_cov or tp < min_sup:
                    break
                cnt = counts[i]
                if cnt == 0:
                    continue
                if cover != parent_cov and not math.isnan(sv[i]):
                    cond = Condition(cb, sv[i], i)
                    self._evaluate_candidate_direct(parent, cond, cover, tp, is_all, best)
                cover -= cnt
                tp -= tps[i]
        else:
            raise AssertionError("SubgroupDiscovery.evaluateNumericRegularSingleBinary() + " + str(op))
        if not is_all:
            self._check_and_log_best(best, parent_cov)

    def _evaluate_numeric_regular_single_binary_coarse(self, parent, cb, value_info):
        ns = self._search_parameters.get_numeric_strategy()
        op = cb.get_operator()
        nr_bins = self._search_parameters.get_nr_bins()
        if nr_bins <= 1:
            return
        b = 1
        col = cb.get_column()
        counts = value_info.itsCounts.tolist()
        tps = value_info.itsTruePositives.tolist()
        sv = col.sorted_floats_view().tolist()
        n = len(counts)
        parent_cov = sum(counts)
        parent_pos = sum(tps)
        is_all = ns == NumericStrategy.NUMERIC_BINS
        best = None if is_all else _BestSubgroups()
        min_cov = self._minimum_coverage
        if op == Operator.BETWEEN:
            last_cover = 0
            last_tp = 0
            f = NEG_INF
            nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
            cover = 0
            tp = 0
            i = 0
            while i < n and b < nr_bins and not self._is_time_to_stop():
                cnt = counts[i]
                if cnt == 0:
                    i += 1
                    continue
                cover += cnt
                tp += tps[i]
                if cover <= nxt or cover < min_cov:
                    i += 1
                    continue
                if cover == parent_cov:
                    break
                nv = sv[i]
                cond = Condition(cb, Interval(f, nv))
                self._evaluate_candidate_direct(parent, cond, cover - last_cover, tp - last_tp, is_all, best)
                last_cover = cover
                last_tp = tp
                f = nv
                b += 1
                nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
                while nxt <= cover - 1:
                    b += 1
                    nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
                i += 1
            if last_cover != parent_cov and float_compare(POS_INF, f) != 0:
                cond = Condition(cb, Interval(f, POS_INF))
                lc = parent_cov - last_cover
                lt = int(parent.get_tertiary_statistic()) - last_tp
                self._evaluate_candidate_direct(parent, cond, lc, lt, is_all, best)
        elif op == Operator.LESS_THAN_OR_EQUAL:
            if n <= nr_bins:
                # fewer unique values than bins, use all cut points
                cover = 0
                tp = 0
                for i in range(n):
                    if self._is_time_to_stop():
                        break
                    cnt = counts[i]
                    if cnt == 0:
                        continue
                    cover += cnt
                    tp += tps[i]
                    if cover < min_cov:
                        continue
                    if cover == parent_cov:
                        break
                    cond = Condition(cb, sv[i], i)
                    self._evaluate_candidate_direct(parent, cond, cover, tp, is_all, best)
            else:
                nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
                cover = 0
                tp = 0
                i = 0
                while b < nr_bins and not self._is_time_to_stop():
                    if i >= n:
                        print("************************i is too big")
                        break
                    cnt = counts[i]
                    if cnt == 0:
                        i += 1
                        continue
                    cover += cnt
                    tp += tps[i]
                    if cover < min_cov or (cover <= nxt and nr_bins - b < n - i - 1):
                        i += 1
                        continue
                    if cover == parent_cov:
                        break
                    cond = Condition(cb, sv[i], i)
                    self._evaluate_candidate_direct(parent, cond, cover, tp, is_all, best)
                    b += 1
                    nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
                    while nxt <= cover - 1:
                        b += 1
                        nxt = get_next_bin_boundary(parent_cov, b, nr_bins)
                    i += 1
        elif op == Operator.GREATER_THAN_OR_EQUAL:
            if n <= nr_bins:
                cover = parent_cov
                tp = parent_pos
                for i in range(n):
                    if self._is_time_to_stop():
                        break
                    if cover < min_cov:
                        break
                    cnt = counts[i]
                    if cnt == 0:
                        continue
                    if cover != parent_cov:
                        cond = Condition(cb, sv[i], i)
                        self._evaluate_candidate_direct(parent, cond, cover, tp, is_all, best)
                    cover -= cnt
                    tp -= tps[i]
            else:
                F32 = np.float32
                bin_size = F32(fdiv(parent_cov, nr_bins))       # long / float -> float
                nxt = int(F32(F32(parent_cov) - bin_size))      # last bin boundary
                cover = parent_cov
                tp = parent_pos
                for i in range(n):
                    if self._is_time_to_stop():
                        break
                    if cover < min_cov:
                        break
                    cnt = counts[i]
                    if cnt == 0:
                        continue
                    if cover <= nxt:
                        cond = Condition(cb, sv[i], i)
                        self._evaluate_candidate_direct(parent, cond, cover, tp, is_all, best)
                        b += 1
                        if b == nr_bins:
                            break
                        # b * aBinSize is float, aParentCoverage - float is float
                        nxt = int(F32(F32(parent_cov) - F32(F32(b) * bin_size)))
                    cover -= counts[i]
                    tp -= tps[i]
        else:
            raise AssertionError("SubgroupDiscovery.evaluateNumericRegularSingleBinaryCoarse() + " + str(op))
        if not is_all:
            self._check_and_log_best(best, parent_cov)

    def _evaluate_candidate_generic(self, parent, added, child_coverage, is_all, best):
        """used by evaluateNumericRegularGeneric(Coarse): members are evaluated"""
        parent_cov = parent.get_coverage()
        child = parent.get_refined_subgroup(added)
        if is_all:
            self._check_and_log(child, parent_cov)
        else:
            self._check_for_best(child, parent_cov, best, False)

    def _evaluate_candidate_direct(self, parent, added, child_coverage, nr_true_positives, is_all, best):
        """used by evaluateNumericRegularSingleBinary(Coarse): direct computation"""
        parent_cov = parent.get_coverage()
        child = direct_computation(parent, added, self._quality_measure, child_coverage, nr_true_positives)
        if is_all:
            self._check_and_log(child, parent_cov)
        else:
            self._check_for_best(child, parent_cov, best, True)

    # ------------------------------------------------------------------
    # numeric intervals
    # ------------------------------------------------------------------

    def _evaluate_numeric_intervals(self, parent, parent_members, ccb):
        if self._search_parameters.get_numeric_strategy() == NumericStrategy.NUMERIC_INTERVALS:
            self._evaluate_numeric_best_interval(parent, parent_members, ccb)
        else:
            self._evaluate_numeric_consecutive_intervals(parent, parent_members, ccb)

    def _evaluate_numeric_best_interval(self, parent, parent_members, ccb):
        """BestInterval, SINGLE_NOMINAL only (brute force over the base intervals)"""
        from .real_base_interval_cross_table import RealBaseIntervalCrossTable
        parent_cov = parent.get_coverage()
        cb = ccb.get(0)
        col = cb.get_column()
        domain = col.get_sorted_values_copy()
        via = col.get_unique_numeric_domain_map_tp(parent_members)
        rbict = RealBaseIntervalCrossTable(parent_cov, int(parent.get_tertiary_statistic()), domain, via)
        # prune split points where adjacent base intervals have the same class distribution
        rbict.aggregate_intervals()
        if rbict.get_nr_split_points() == 0:
            return
        best_sg = self._evaluate_numeric_best_interval_brute_force_mm(parent, cb, rbict)
        if best_sg is None:
            return
        best_tp = int(best_sg.get_tertiary_statistic())
        best_fp = best_sg.get_coverage() - best_tp
        best_interval = best_sg.get_conditions().get(best_sg.get_depth() - 1).get_numeric_interval()
        best_q = best_sg.get_measure_value()
        child_cov = best_tp + best_fp
        if child_cov == parent_cov:
            self._skip_count += 1
            if BEST_INTERVAL_NO_REFINEMENT_SKIP:
                return
        cond = Condition(cb, best_interval)
        child = direct_computation(parent, cond, best_q, child_cov, best_tp)
        self._check_and_log(child, parent_cov)

    def _evaluate_numeric_best_interval_brute_force_mm(self, parent, cb, rbict):
        unset = -2147483648
        use_infinite = -1
        best_q = NEG_INF
        best_lo = unset
        best_hi = unset
        best_tp = unset
        best_cov = unset
        nr_sp = rbict.get_nr_split_points()
        parent_cov = parent.get_coverage()
        parent_tps = int(parent.get_tertiary_statistic())
        qm = self._quality_measure
        min_cov = self._minimum_coverage
        max_cov = self._maximum_coverage
        pos = [rbict.get_positive_count(i) for i in range(nr_sp)]
        neg = [rbict.get_negative_count(i) for i in range(nr_sp)]
        head_cover = 0
        head_tp = 0
        for i in range(nr_sp):
            pi = pos[i]
            head_cover += pi + neg[i]
            head_tp += pi
            # from this split point to Infinity
            if is_useful_for_result_set(parent_cov - head_cover, parent_cov, min_cov, max_cov):
                q = to_f32(qm.calculate(parent_tps - head_tp, parent_cov - head_cover))
                if q > best_q:
                    best_q = q
                    best_lo = i + 1
                    best_hi = use_infinite
                    best_tp = parent_tps - head_tp
                    best_cov = parent_cov - head_cover
            # from -Infinity to this split point
            if is_useful_for_result_set(head_cover, parent_cov, min_cov, max_cov):
                q = float('nan') if head_cover == 0 else to_f32(qm.calculate(head_tp, head_cover))
                if q > best_q:
                    best_q = q
                    best_lo = use_infinite
                    best_hi = i
                    best_tp = head_tp
                    best_cov = head_cover
            # from this split point to all later ones (i is the exclusive left end)
            cover = 0
            tp = 0
            lo = i + 1
            for j in range(lo, nr_sp):
                pj = pos[j]
                cover += pj + neg[j]
                tp += pj
                if is_useful_for_result_set(cover, parent_cov, min_cov, max_cov):
                    q = to_f32(qm.calculate(tp, cover))
                    if q > best_q:
                        best_q = q
                        best_lo = lo
                        best_hi = j
                        best_tp = tp
                        best_cov = cover
        if best_cov == unset:
            return None
        sp = rbict.get_split_points()
        lo_v = NEG_INF if best_lo < 0 else sp[best_lo - 1]
        hi_v = POS_INF if best_hi < 0 else sp[best_hi]
        cond = Condition(cb, Interval(lo_v, hi_v))
        return direct_computation(parent, cond, best_q, best_cov, best_tp)

    def _evaluate_numeric_best_interval_linear(self, parent, cb, rbict):
        """linear BestInterval with convex hulls, debug only in java"""
        from .convex_hull import ConvexHull
        best_q = NEG_INF
        best_fp = -2147483648
        best_tp = -2147483648
        best_interval = Interval(NEG_INF, POS_INF)
        nbi = rbict.get_nr_base_intervals()
        hulls = [None] * nbi
        a_p = 0
        a_n = 0
        for l in range(rbict.get_nr_split_points()):
            a_p += rbict.get_positive_count(l)
            a_n += rbict.get_negative_count(l)
            hulls[l] = ConvexHull(a_n, a_p, rbict.get_split_point(l), NEG_INF)
        hulls[nbi - 1] = ConvexHull(rbict.get_negative_count(), rbict.get_positive_count(), POS_INF, NEG_INF)
        k = nbi
        while k > 1:
            for l in range(0, k - 1, 2):
                mink = hulls[l].minkowski_difference(hulls[l + 1], True)
                for side in range(2):
                    for i in range(mink.get_size(side)):
                        if self._is_time_to_stop():
                            break
                        if side == 1 and (i == 0 or i == mink.get_size(side) - 1):
                            continue
                        c = mink.get_point(side, i)
                        q = self._quality_measure.calculate(c.y, c.x + c.y)
                        if q > best_q:
                            best_q = q
                            best_fp = c.x
                            best_tp = c.y
                            best_interval = Interval(c.get_label2(), c.label1)
            for l in range(0, k - 1, 2):
                hulls[l // 2] = hulls[l].concatenate(hulls[l + 1])
            if k % 2 == 1:
                hulls[k // 2] = hulls[k - 1]
            k = (k + 1) // 2
        cond = Condition(cb, best_interval)
        return direct_computation(parent, cond, best_q, best_tp + best_fp, best_tp)

    def _evaluate_numeric_consecutive_intervals(self, parent, parent_members, ccb):
        """deprecated vm-consecutive-all / vm-consecutive-best strategies"""
        ns = self._search_parameters.get_numeric_strategy()
        parent_cov = parent.get_coverage()
        cb = ccb.get(0)
        col = cb.get_column()
        is_all = ns == NumericStrategy.NUMERIC_VIKAMINE_CONSECUTIVE_ALL
        best_sg = None
        nr_split_points = self._search_parameters.get_nr_bins() - 1
        if nr_split_points <= 0:
            return
        intervals = col.get_unique_split_points_bounded(parent_members, parent_cov, nr_split_points)
        if len(intervals) <= 1:
            return
        for iv, cnt in intervals:
            if self._is_time_to_stop():
                break
            if cnt < self._minimum_coverage:
                continue
            if cnt == parent_cov:
                break
            cond = Condition(cb, iv)
            child = parent.get_refined_subgroup(cond)
            if is_all:
                self._check_and_log(child, parent_cov)
            elif self._is_valid_and_best(child, parent_cov, best_sg):
                best_sg = child
        if not is_all and best_sg is not None:
            self._check_and_log(best_sg, parent_cov)

    def _is_valid_and_best(self, new_sg, old_coverage, best_sg):
        """XXX replaced by check_for_best() in java, still used by the vm-* code"""
        cov = new_sg.get_coverage()
        if cov >= self._minimum_coverage and cov < old_coverage and cov <= self._maximum_coverage:
            q = self._evaluate_candidate(new_sg)
            if best_sg is None or q > best_sg.get_measure_value():
                new_sg.set_measure_value(q)
                return True
        return False

    # ------------------------------------------------------------------
    # best / check
    # ------------------------------------------------------------------

    def _check_for_best(self, child, parent_coverage, best, quality_computed):
        """NOTE (java) for <= the smallest of equal scores wins, for >= the largest"""
        cov = child.get_coverage()
        for_cs = is_useful_for_candidate_set(cov, parent_coverage, self._minimum_coverage, child.get_depth(),
                                             self._search_parameters.get_search_depth())
        for_rs = is_useful_for_result_set(cov, parent_coverage, self._minimum_coverage, self._maximum_coverage)
        if not for_cs and not for_rs:
            return
        if not quality_computed:
            child.set_measure_value(self._evaluate_candidate(child))
        q = child.get_measure_value()
        if for_cs and (best.for_candidate_set is None or q > best.for_candidate_set.get_measure_value()):
            best.for_candidate_set = child
        if for_rs and (best.for_result_set is None or q > best.for_result_set.get_measure_value()):
            best.for_result_set = child

    def _check_and_log_best(self, best, parent_coverage):
        if best.for_result_set is not None:
            self._check_and_log(best.for_result_set, parent_coverage)
        if USE_SINGLE_BEST_RESULT_LIKE_BEFORE:
            return
        if best.for_candidate_set is not None and best.for_result_set is not best.for_candidate_set:
            self._check_and_log(best.for_candidate_set, parent_coverage)

    def _check_and_log(self, child, parent_coverage):
        """the only place that adds to the result set and candidate queue"""
        self._set_title(child)
        cov = child.get_coverage()
        depth = child.get_depth()
        sp = self._search_parameters
        for_cs = is_useful_for_candidate_set(cov, parent_coverage, self._minimum_coverage, depth, sp.get_search_depth())
        for_rs = is_useful_for_result_set(cov, parent_coverage, self._minimum_coverage, self._maximum_coverage)
        is_valid = for_cs or for_rs
        if is_valid:
            last_added = child.get_conditions().get(depth - 1).get_column().get_type()
            is_last_numeric = last_added == AttributeType.NUMERIC
            direct = self._is_direct_single_binary()
            if last_added == AttributeType.BINARY and direct:
                quality = to_f32(child.get_measure_value())
            elif is_last_numeric and direct:
                quality = to_f32(child.get_measure_value())
            elif is_last_numeric and sp.get_numeric_strategy() in _NUMERIC_BEST:
                quality = to_f32(child.get_measure_value())
            elif last_added == AttributeType.NOMINAL and sp.get_nominal_sets():
                quality = to_f32(child.get_measure_value())
            else:
                quality = self._evaluate_candidate(child)
                child.set_measure_value(quality)
            # minimum support, in SINGLE_NOMINAL the tertiary statistic is the nr of positives
            if sp.get_target_type() == TargetType.SINGLE_NOMINAL and child.get_tertiary_statistic() < self._minimum_support:
                for_rs = False
                for_cs = False
            for_rs = for_rs and (self._ignore_quality_minimum or quality > self._quality_measure_minimum)
            if for_rs:
                for_rs = self._result.has_potential(quality)
            if for_cs or for_rs:
                candidate = Candidate(child) if for_cs else None
                if for_rs:
                    self._result.add(child)
                if for_cs:
                    self._candidate_queue.add(candidate)
        # free the memory, members are re-evaluated when needed
        child.kill_members()
        count = self._candidate_count
        self._candidate_count += 1
        if is_valid:
            self._log_candidate_addition(child, count)

    def _log_candidate_addition(self, subgroup, count):
        if NO_CANDIDATE_LOG:
            return
        log.log_command_line("candidate " + str(subgroup.get_conditions()) + " size: " + str(subgroup.get_coverage())
                             + "\n  subgroup nr. " + str(count) + "; quality " + double_to_string(subgroup.get_measure_value()))

    # ------------------------------------------------------------------
    # evaluation per target type
    # ------------------------------------------------------------------

    def _evaluate_candidate(self, child):
        t = self._search_parameters.get_target_type()
        if t == TargetType.SINGLE_NOMINAL:
            return self._evaluate_single_nominal(child)
        if t == TargetType.SINGLE_NUMERIC:
            return self._evaluate_single_numeric(child)
        if t == TargetType.MULTI_NUMERIC:
            return self._evaluate_multi_numeric(child)
        if t == TargetType.DOUBLE_REGRESSION:
            return self._evaluate_double_regression(child)
        if t == TargetType.DOUBLE_CORRELATION:
            return self._evaluate_double_correlation(child)
        if t == TargetType.DOUBLE_BINARY:
            return self._evaluate_double_binary(child)
        if t == TargetType.SCAPE:
            return self._evaluate_scape(child)
        if t == TargetType.MULTI_LABEL:
            return self._evaluate_multi_label(child)
        if t == TargetType.LABEL_RANKING:
            return self._evaluate_label_ranking(child)
        raise AssertionError(t)

    def _evaluate_single_nominal(self, child):
        cov = child.get_coverage()
        members = child.members_view()
        hb = int(np.count_nonzero(members & self._binary_target))
        qm = self._search_parameters.get_quality_measure()
        if qm not in (QM.PROP_SCORE_WRACC, QM.PROP_SCORE_RATIO):
            quality = to_f32(self._quality_measure.calculate(hb, cov))
        else:
            from .propensity_score import PropensityScore, LOGISTIC_REGRESSION
            ps = PropensityScore(child, self._binary_target, self._local_knowledge, self._global_knowledge, LOGISTIC_REGRESSION)
            quality = calculate_propensity_based(qm, hb, cov, self._nr_rows, ps.get_propensity_score_sum())
        child.set_secondary_statistic(div(hb, float(cov)))
        child.set_tertiary_statistic(hb)
        return quality

    def _evaluate_single_numeric(self, child):
        members = child.members_view()
        if not TEMPORARY_CODE:
            qm = self._search_parameters.get_quality_measure()
            req = QM.required_stats(qm)
            stats = self._numeric_target.get_statistics(self._selection, members, qm == QM.MMAD, Stat.COMPL in req)
            pdf = None
            if Stat.PDF in req:
                from .probability_density_function import (ProbabilityDensityFunction, ProbabilityDensityFunction2,
                                                           USE_ProbabilityDensityFunction2)
                print("#Subgroup: '%s' (size = %d)" % (child, child.get_coverage()))
                if not USE_ProbabilityDensityFunction2:
                    pdf = ProbabilityDensityFunction(self._quality_measure.get_probability_density_function(), members)
                else:
                    pdf = ProbabilityDensityFunction2(self._quality_measure.get_probability_density_function(), members)
                pdf.smooth()
            quality = self._quality_measure.calculate(stats, pdf)
            child.set_secondary_statistic(stats.get_subgroup_average())
            # n-1 like the t-statistic
            child.set_tertiary_statistic(jsqrt(div(stats.get_subgroup_sum_squared_deviations(), child.get_coverage() - 1.0)))
        else:
            from .column_statistics import Statistics
            from .probability_mass_function_nd import ProbabilityMassFunctionND
            cov = child.get_coverage()
            print("#Subgroup: '%s' (size = %d)" % (child, cov))
            pmf = ProbabilityMassFunctionND(self._quality_measure.get_probability_density_function(), members)
            quality = self._quality_measure.calculate(Statistics(cov, -1, -1, -1, -1), pmf)
            child.set_secondary_statistic(cov)
            child.set_tertiary_statistic(self._nr_rows - cov)
        return quality

    def _evaluate_multi_numeric(self, child):
        # for now always test against the complement
        quality = self.pdf_nd.get_density_difference(child.members_view(), True, self._search_parameters.get_quality_measure())
        child.set_secondary_statistic(child.get_coverage())
        child.set_tertiary_statistic(self._nr_rows - child.get_coverage())
        return quality

    def _evaluate_double_regression(self, child):
        from .regression_measure import RegressionMeasure
        members = child.members_view()
        q = self._base_rm.get_quality_measure()
        if q in (QM.REGRESSION_SSD_COMPLEMENT, QM.REGRESSION_SSD_DATASET, QM.REGRESSION_FLATNESS, QM.REGRESSION_SSD_4):
            rm = RegressionMeasure.from_base(self._base_rm, members)
            quality = rm.get_evaluation_measure_value()
            child.set_secondary_statistic(rm.get_slope())
            child.set_tertiary_statistic(rm.get_intercept())
            return to_f32(quality)
        # COOKS_DISTANCE is disabled in the java code as well
        raise AssertionError(q)

    def _evaluate_double_correlation(self, child):
        from .correlation_measure import CorrelationMeasure
        members = child.members_view()
        cm = CorrelationMeasure.copy_of(self._base_cm)
        n = self._primary_column.size()
        idx = np.flatnonzero(members[:n])
        cm.add_observations(self._primary_column.floats_view()[idx], self._secondary_column.floats_view()[idx])
        child.set_secondary_statistic(cm.get_correlation())
        child.set_tertiary_statistic(cm.compute_correlation_distance())
        return to_f32(cm.get_evaluation_measure_value())

    def _evaluate_double_binary(self, child):
        F = np.float32
        members = child.members_view()
        primary = self._primary_column.get_binaries()
        secondary = self._secondary_column.get_binaries()
        size_a = int(np.count_nonzero(primary))
        size_b = self._nr_rows - size_a
        with np.errstate(all='ignore'):
            target_a = F(F(int(np.count_nonzero(secondary & primary))) / F(size_a))
            target_b = F(F(int(np.count_nonzero(secondary & ~primary))) / F(size_b))
            sub_a = members[:primary.shape[0]] & primary
            count_a = int(np.count_nonzero(sub_a))
            if count_a == 0:   # FIXME (java)
                child.set_secondary_statistic(0)
                return 0.0
            count_as = int(np.count_nonzero(sub_a & secondary))
            sg_target_a = F(F(count_as) / F(count_a))
            child.set_tertiary_statistic(float(sg_target_a))
            sub_b = members[:primary.shape[0]] & ~primary
            count_b = int(np.count_nonzero(sub_b))
            if count_b == 0:
                child.set_secondary_statistic(0)
                return 0.0
            count_bs = int(np.count_nonzero(sub_b & secondary))
            sg_target_b = F(F(count_bs) / F(count_b))
            rel_a = F(F(sg_target_a - target_a) / target_a)
            child.set_secondary_statistic(float(np.sign(rel_a)) if not math.isnan(float(rel_a)) else float('nan'))
            wracc_a = F(F(F(count_as) / F(size_a)) - F(F(target_a * F(count_a)) / F(size_a)))
            wracc_b = F(F(F(count_bs) / F(size_b)) - F(F(target_b * F(count_b)) / F(size_b)))
            qm = self._search_parameters.get_quality_measure()
            if qm == QM.RELATIVE_WRACC:
                return float(F(wracc_a / wracc_b)) if wracc_b != 0 else float(np.finfo(np.float32).max)
            if qm == QM.ABSOLUTE_WRACC:
                return float(F(wracc_a - wracc_b))
            rel_b = F(F(sg_target_b - target_b) / target_b)
            if qm == QM.RELATIVE_RISK:
                return float(F(rel_a / rel_b))
            if qm == QM.ABSOLUTE_RISK:
                return float(F(rel_a - rel_b))
        return 0.0   # should not happen

    def _evaluate_scape(self, child):
        cov = child.get_coverage()
        members = child.members_view()
        hb = int(np.count_nonzero(members[:self._primary_column.size()] & self._primary_column.binaries_view()))
        quality = self._quality_measure.calculate(members.copy(), cov, hb)
        child.set_secondary_statistic(hb)
        child.set_tertiary_statistic(cov - hb)
        return quality

    def _evaluate_multi_label(self, child):
        from .bayesian import Bayesian
        bt = self._binary_table.select_rows(child.get_members())
        bayes = Bayesian(bt, self._targets)
        bayes.climb()
        child.set_dag(bayes.get_dag())
        quality = self._quality_measure.calculate(child)
        child.set_secondary_statistic(self._quality_measure.calculate_edit_distance(child.get_dag()))
        child.set_tertiary_statistic(calculate_entropy(self._nr_rows, child.get_coverage()))
        return quality

    def _evaluate_label_ranking(self, child):
        cov = child.get_coverage()
        lrm = self._target_rankings.get_average_ranking_matrix(child)
        quality = self._quality_measure.compute_label_ranking_distance(cov, lrm)
        child.set_label_ranking(self._target_rankings.get_average_ranking(child))
        child.set_label_ranking_matrix(lrm)
        return quality

    # ------------------------------------------------------------------
    # info
    # ------------------------------------------------------------------

    def get_number_of_subgroups(self):
        return len(self._result)

    def get_result(self):
        return self._result

    def get_quality_measure(self):
        return self._quality_measure

    def get_search_parameters(self):
        return self._search_parameters

    def get_regression_measure_base(self):
        """base RegressionMeasure for DOUBLE_REGRESSION, else None"""
        return self._base_rm

    def get_candidate_count(self):
        return self._candidate_count


def _i32(x):
    from .java_compat import int32
    return int32(x)


def _prepare_data(binary_target, columns):
    """sorted values / shared domains, do not change the data until mining ends"""
    for c in columns:
        t = c.get_type()
        if t == AttributeType.NOMINAL:
            c.build_shared_domain()
        elif t == AttributeType.NUMERIC:
            log.log_command_line(c.get_name())
            c.build_sorted(binary_target)
        elif t == AttributeType.BINARY:
            pass
        else:
            raise AssertionError("SubgroupDiscovery.prepareData(): " + str(t))


def _finish_title(main_window, elapsed, nr_candidates):
    if main_window is None:
        return
    minutes = elapsed // 60000
    seconds = (elapsed % 60000) / 1000.0
    msg = "" if minutes == 0 else " %d minute%s and" % (minutes, "" if minutes == 1 else "s")
    msg = "%s %.3f seconds" % (msg, seconds)
    msg = "Finished: evaluated %s candidate%s in%s." % (_german_number(nr_candidates), "" if nr_candidates == 1 else "s", msg)
    main_window.set_mining_title(msg)
