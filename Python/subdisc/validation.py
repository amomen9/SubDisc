"""Statistical validation: qualities of random subsets, random descriptions or
swap-randomised targets, to build a distribution of 'chance' qualities."""

import time

import numpy as np

from . import log
from . import condition_list_builder as clb
from .attribute_type import AttributeType
from .condition import Condition, UNINITIALISED_SORT_INDEX
from .condition_base import ConditionBase
from .java_compat import JavaRandom, double_to_string, div
from .operators import Operator
from .qm import QM
from .stat_enum import Stat
from .subgroup import Subgroup
from .target_type import TargetType

# names used by the gui (RandomQualitiesWindow)
RANDOM_SUBSETS = "Random subsets"
RANDOM_DESCRIPTIONS = "Random descriptions"
SWAP_RANDOMIZATION = "Swap-randomization"
DEFAULT_AMOUNT = "100"


def is_valid_random_qualities_setup(setup):
    """(method, amount) with a known method and an amount > 1"""
    from .java_compat import parse_int
    try:
        return (setup is not None and len(setup) == 2 and setup[0] is not None and setup[1] is not None
                and setup[0] in (RANDOM_SUBSETS, RANDOM_DESCRIPTIONS, SWAP_RANDOMIZATION)
                and not (parse_int(setup[1]) <= 1))
    except ValueError:
        return False


class Validation:
    def __init__(self, search_parameters, table, selection, quality_measure):
        """quality_measure is only used for random subsets / descriptions"""
        self._search_parameters = search_parameters
        self._target_concept = search_parameters.get_target_concept()
        self._table = table
        self._selection = selection
        self._quality_measure = quality_measure

    def get_qualities(self, setup):
        """setup = (method, nr_repetitions) as the RandomQualitiesWindow gives it"""
        if not is_valid_random_qualities_setup(setup):
            return None
        method = setup[0]
        reps = int(setup[1])
        if method == RANDOM_SUBSETS:
            return self.get_random_qualities(True, reps)
        if method == RANDOM_DESCRIPTIONS:
            return self.get_random_qualities(False, reps)
        if method == SWAP_RANDOMIZATION:
            return self._swap_randomization(reps)
        return None

    @staticmethod
    def is_valid_random_qualities_target_type(t):
        return t in (TargetType.SINGLE_NOMINAL, TargetType.SINGLE_NUMERIC, TargetType.DOUBLE_REGRESSION,
                     TargetType.DOUBLE_CORRELATION, TargetType.DOUBLE_BINARY, TargetType.MULTI_LABEL,
                     TargetType.LABEL_RANKING)

    def get_random_qualities(self, for_subgroups, nr_repetitions):
        min_cov = self._search_parameters.get_minimum_coverage()
        rnd = JavaRandom(10)
        depth = self._search_parameters.get_search_depth()
        t = self._target_concept.get_target_type()
        if t == TargetType.SINGLE_NOMINAL:
            return self._single_nominal(for_subgroups, nr_repetitions, min_cov, rnd, depth)
        if t == TargetType.SINGLE_NUMERIC:
            return self._single_numeric(for_subgroups, nr_repetitions, min_cov, rnd, depth)
        if t == TargetType.DOUBLE_REGRESSION:
            return self._double_regression(for_subgroups, nr_repetitions, min_cov, rnd, depth)
        if t in (TargetType.DOUBLE_CORRELATION, TargetType.DOUBLE_BINARY):
            return self._double_correlation(for_subgroups, nr_repetitions, min_cov, rnd, depth)
        if t == TargetType.MULTI_LABEL:
            return self._multi_label(for_subgroups, nr_repetitions, min_cov, rnd, depth)
        if t == TargetType.LABEL_RANKING:
            return self._label_ranking(for_subgroups, nr_repetitions, min_cov, rnd, depth)
        raise AssertionError(t)

    def _next(self, for_subgroups, min_cov, rnd, depth):
        if for_subgroups:
            return self._get_valid_subgroup(min_cov, rnd)
        return self._get_valid_subgroup_conditions(depth, min_cov, rnd)

    def _single_nominal(self, for_subgroups, reps, min_cov, rnd, depth):
        # FIXME (java) why is this here, the binary target is known already
        target = self._target_concept.get_primary_target()
        cb = ConditionBase(target, Operator.EQUALS)
        value = self._target_concept.get_target_value()
        if target.get_type() == AttributeType.NOMINAL:
            cond = Condition(cb, value)
        elif target.get_type() == AttributeType.BINARY:
            if not AttributeType.is_valid_binary_value(value):
                raise ValueError(str(value) + " is not a valid BINARY value")
            cond = Condition(cb, AttributeType.is_valid_binary_true_value(value))
        else:
            raise AssertionError(target.get_type())
        b = target.evaluate(np.ones(self._table.get_nr_rows(), dtype=bool), cond)
        out = []
        for _ in range(reps):
            sg = self._next(for_subgroups, min_cov, rnd, depth)
            hb = int(np.count_nonzero(sg.members_view() & b))
            out.append(self._quality_measure.calculate(hb, sg.get_coverage()))
        return out

    def _single_numeric(self, for_subgroups, reps, min_cov, rnd, depth):
        from .probability_density_function import (ProbabilityDensityFunction, ProbabilityDensityFunction2,
                                                   USE_ProbabilityDensityFunction2)
        out = []
        target = self._target_concept.get_primary_target()
        for _ in range(reps):
            sg = self._next(for_subgroups, min_cov, rnd, depth)
            members = sg.members_view()
            qm = self._search_parameters.get_quality_measure()
            stats = target.get_statistics(None, members, qm == QM.MMAD, Stat.COMPL in QM.required_stats(qm))
            base = self._quality_measure.get_probability_density_function()
            if not USE_ProbabilityDensityFunction2:
                pdf = ProbabilityDensityFunction(base, members)
            else:
                pdf = ProbabilityDensityFunction2(base, members)
            pdf.smooth()
            out.append(self._quality_measure.calculate(stats, pdf))
        return out

    def _double_regression(self, for_subgroups, reps, min_cov, rnd, depth):
        from .regression_measure import RegressionMeasure
        p = self._target_concept.get_primary_target()
        s = self._target_concept.get_secondary_target()
        base = RegressionMeasure(self._search_parameters.get_quality_measure(), p, s)
        out = []
        for _ in range(reps):
            sg = self._next(for_subgroups, min_cov, rnd, depth)
            out.append(RegressionMeasure.from_base(base, sg.members_view()).get_evaluation_measure_value())
        return out

    def _double_correlation(self, for_subgroups, reps, min_cov, rnd, depth):
        from .correlation_measure import CorrelationMeasure
        p = self._target_concept.get_primary_target()
        s = self._target_concept.get_secondary_target()
        base = CorrelationMeasure(self._search_parameters.get_quality_measure(), p, s)
        out = []
        for _ in range(reps):
            sg = self._next(for_subgroups, min_cov, rnd, depth)
            cm = CorrelationMeasure.copy_of(base)
            # NOTE java loops with nextSetBit(k) here, which never ends;
            # nextSetBit(k + 1) is what was meant
            idx = np.flatnonzero(sg.members_view()[:p.size()])
            cm.add_observations(p.floats_view()[idx], s.floats_view()[idx])
            out.append(cm.get_evaluation_measure_value())
        return out

    def _multi_label(self, for_subgroups, reps, min_cov, rnd, depth):
        from .bayesian import Bayesian
        from .binary_table import BinaryTable
        base_table = BinaryTable(self._table, self._target_concept.get_multi_targets())
        bayes = Bayesian(base_table)
        bayes.climb()
        out = []
        for i in range(reps):
            sg = self._next(for_subgroups, min_cov, rnd, depth)
            bt = base_table.select_rows(sg.get_members())
            bayes = Bayesian(bt)
            bayes.climb()
            sg.set_dag(bayes.get_dag())
            q = self._quality_measure.calculate(sg)
            out.append(q)
            # XXX (java) original code only did this for Condition
            if not for_subgroups:
                log.log_command_line("%d,%d,%s" % (i + 1, sg.get_coverage(), double_to_string(q)))
        return out

    def _label_ranking(self, for_subgroups, reps, min_cov, rnd, depth):
        from .quality_measure import QualityMeasure
        target = self._target_concept.get_primary_target()
        lr = target.get_average_ranking(None)
        lrm = target.get_average_ranking_matrix(None)
        qm = QualityMeasure(self._search_parameters.get_quality_measure(), self._table.get_nr_rows(), lr, lrm)
        out = []
        # temp (java): the qualities also go to output.txt
        try:
            br = open("output.txt", "w", encoding="utf-8", newline="")
        except OSError:
            br = None
        try:
            for _ in range(reps):
                sg = self._next(for_subgroups, min_cov, rnd, depth)
                sg_lrm = target.get_average_ranking_matrix(sg)
                q = qm.compute_label_ranking_distance(sg.get_coverage(), sg_lrm)
                out.append(q)
                log.log_command_line("qual: " + double_to_string(q))
                if br is not None:
                    br.write(double_to_string(q) + "\r")
        finally:
            if br is not None:
                br.close()
        return out

    def _get_valid_subgroup(self, min_cov, rnd):
        """random subset, a new random size every time"""
        n = self._table.get_nr_rows()
        while True:
            size = int(rnd.next_double() * n)
            if not (size < min_cov or size == n):
                break
        return Subgroup(None, self._table.get_random_bit_set(size))

    def _get_valid_subgroup_conditions(self, depth, min_cov, rnd):
        """random description"""
        n = self._table.get_nr_rows()
        while True:
            cl = self._get_random_condition_list(depth, rnd)
            members = self._table.evaluate(cl)
            size = int(np.count_nonzero(members))
            if not (size < min_cov or size == n):
                break
        log.log_command_line(str(cl))
        return Subgroup(None, members)

    def _swap_randomization(self, reps):
        from .subgroup_discovery import SubgroupDiscovery
        mem = log.COMMANDLINELOG
        qualities = [0.0] * reps
        tc = self._target_concept
        t = tc.get_target_type()
        table = self._table
        sp = self._search_parameters
        sel = self._selection
        cols = table.get_columns()

        def loop(make):
            i = 0
            while i < reps:
                table.swap_randomize_target(tc)
                i = self._run_srsd(make(), qualities, i)
                i += 1

        if t == TargetType.SINGLE_NOMINAL:
            copy = tc.get_primary_target().copy()
            positives = tc.get_primary_target().count_values(tc.get_target_value(), sel)
            loop(lambda: SubgroupDiscovery.single_nominal(sp, table, sel, positives, None))
            tc.set_primary_target(copy)
            cols[copy.get_index()] = copy
        elif t == TargetType.SINGLE_NUMERIC:
            copy = tc.get_primary_target().copy()
            avg = tc.get_primary_target().get_average(sel)
            loop(lambda: SubgroupDiscovery.single_numeric(sp, table, sel, avg, None))
            tc.set_primary_target(copy)
            cols[copy.get_index()] = copy
        elif t in (TargetType.DOUBLE_REGRESSION, TargetType.DOUBLE_CORRELATION, TargetType.DOUBLE_BINARY):
            pc = tc.get_primary_target().copy()
            sc = tc.get_secondary_target().copy()
            reg = t == TargetType.DOUBLE_REGRESSION
            loop(lambda: SubgroupDiscovery.double_target(sp, table, sel, reg, None))
            tc.set_primary_target(pc)
            cols[pc.get_index()] = pc
            tc.set_secondary_target(sc)
            cols[sc.get_index()] = sc
        elif t == TargetType.MULTI_LABEL:
            copies = [c.copy() for c in tc.get_multi_targets()]
            # swap randomisation before creating the new SubgroupDiscovery
            loop(lambda: SubgroupDiscovery.multi_label(sp, table, sel, None))
            tc.set_multi_targets(copies)
            for c in copies:
                cols[c.get_index()] = c
        elif t == TargetType.LABEL_RANKING:
            copy = tc.get_primary_target().copy()
            loop(lambda: SubgroupDiscovery.label_ranking(sp, None, table, sel))
            tc.set_primary_target(copy)
            cols[copy.get_index()] = copy
        else:
            raise AssertionError(t)
        log.COMMANDLINELOG = mem
        return qualities

    def get_sign_with_swap_rand(self, reps):
        """5% significance of swap randomisation"""
        from .normal_distribution import NormalDistribution
        q = self._swap_randomization(reps)
        return NormalDistribution(q).get_five_percent_significance()

    def _run_srsd(self, sd, qualities, repetition):
        """NOTE swapRandomizeTarget() must run before creating sd"""
        sd.ignore_quality_minimum()
        log.COMMANDLINELOG = False
        sd.mine(int(time.time() * 1000), self._search_parameters.get_nr_threads())
        log.COMMANDLINELOG = True
        result = sd.get_result()
        if len(result) == 0:
            return repetition - 1   # no subgroups, try again
        qualities[repetition] = result.get_best_score()
        log.log_command_line(str(repetition + 1) + ", " + double_to_string(qualities[repetition]))
        return repetition

    def _get_random_condition_list(self, depth, rnd):
        F = np.float32
        d = 1 + rnd.next_int(depth)
        cl = clb.empty_list()
        nr_columns = self._table.get_nr_columns()
        for _ in range(d):
            while True:
                col = self._table.get_column(rnd.next_int(nr_columns))
                if not (self._target_concept.is_target_attribute(col) or not col.get_is_enabled()):
                    break
            t = col.get_type()
            if t == AttributeType.NOMINAL:
                domain = col.get_domain()
                idx = int(rnd.next_double() * len(domain))
                cond = Condition(ConditionBase(col, Operator.EQUALS), domain[idx])
            elif t == AttributeType.NUMERIC:
                op = Operator.LESS_THAN_OR_EQUAL if rnd.next_boolean() else Operator.GREATER_THAN_OR_EQUAL
                mn = F(col.get_min())
                mx = F(col.get_max())
                rng = F(mx - mn)
                with np.errstate(all='ignore'):
                    v = F(F(mn + F(F(0.1) * rng)) + F(F(F(0.8) * rng) * F(rnd.next_float())))
                # value may not occur in the Column, so no sort index
                cond = Condition(ConditionBase(col, op), float(v), UNINITIALISED_SORT_INDEX)
            elif t == AttributeType.BINARY:
                cond = Condition(ConditionBase(col, Operator.EQUALS), rnd.next_boolean())
            else:
                raise AssertionError(t)
            cl = clb.create_list(cl, cond)
        return cl

    @staticmethod
    def perform_regression_test(qualities, subgroup_set):
        one = _regression_test(qualities, 1, subgroup_set)
        ten = 3.141592653589793
        if len(subgroup_set) >= 10:
            ten = _regression_test(qualities, 10, subgroup_set)
        return [one, ten]

    @staticmethod
    def compute_empirical_p_value(qualities, subgroup_set):
        k = 1   # hardcoded
        it = iter(subgroup_set)
        top = 0.0
        for _ in range(k):
            top += next(it).get_measure_value()
        top /= k
        count = sum(1 for q in qualities if q > top)
        qualities.sort()
        log.log_command_line("Empirical p-value: " + double_to_string(count / float(len(qualities))))
        return count / float(len(qualities))


def _regression_test(qualities, k, subgroup_set):
    # sic: java adds the quality of ALL subgroups, then divides by k
    top = 0.0
    for s in subgroup_set:
        top += s.get_measure_value()
    top /= k
    n = len(qualities)
    copy = sorted(qualities)
    mn = min(copy[0], top)
    mx = max(copy[n - 1], top)
    x_bar = 0.5
    y_bar = 0.0
    for i in range(n):
        copy[i] = div(copy[i] - mn, mx - mn)
        y_bar += copy[i]
    top = div(top - mn, mx - mn)
    y_bar = (y_bar + top) / (float(n) + 1)
    xx_bar = 0.25
    xy_bar = 0.5 * (top - y_bar)
    xs = [float(i) / float(n) for i in range(n)]
    for i in range(n):
        log.log_command_line(double_to_string(xs[i]) + "\t" + double_to_string(copy[i]))
    for i in range(n):
        xx_bar += (xs[i] - x_bar) * (xs[i] - x_bar)
        xy_bar += (xs[i] - x_bar) * (copy[i] - y_bar)
    beta1 = div(xy_bar, xx_bar)
    beta0 = y_bar - beta1 * x_bar
    log.log_command_line("Fitted regression line: y = " + double_to_string(beta1) + " * x + " + double_to_string(beta0))
    score = top - beta1 - beta0
    log.log_command_line("Regression test score: " + double_to_string(score))
    return score
