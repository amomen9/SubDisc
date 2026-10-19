"""The result set: Subgroups sorted best first, optionally capped in size.

In a nominal target setting a ROCList can be built from it for the ROC
window. Only add() was thread safe in java; the python port runs the
search in one thread anyway.
"""

import math

import numpy as np

from . import log
from .java_compat import div, double_to_string
from .search_strategy import SearchStrategy
from .tree_set import TreeSet

USE_OLD_COVER_BASED_SUBGROUP_SELECTION = False
COVER_BASED_SUBGROUP_SELECTION_TOP_K = 100
INT_MAX = 2147483647


class SubgroupSet(TreeSet):
    ROC_HEADER = ("ID", "FPR", "TPR", "Conditions")
    CHECK_FOR_ALTERNATIVE_DESCRIPTIONS = False

    def __init__(self, size=0, selection=None, nr_rows=0, binary_target=None, nominal_target_setting=None, comparator=None):
        """SubgroupSet(size, selection, total_coverage[, binary_target])
        size <= 0 means no maximum size."""
        TreeSet.__init__(self, comparator)
        if nominal_target_setting is None:
            nominal_target_setting = binary_target is not None
        self._maximum_size = INT_MAX if size <= 0 else size
        self._nr_rows = nr_rows
        self._nominal_target_setting = nominal_target_setting
        self._binary_target = binary_target
        self._roc_list = None
        self._lowest_score = float('nan')
        self._joint_entropy = float('nan')
        self._binary_table = None
        if comparator is not None:
            # CAUC(Heavy) setting only (see Process)
            self._all_data = None
            self._total_coverage = 0
            self._nr_rows = -1
            return
        if selection is None:
            self._all_data = np.ones(nr_rows, dtype=bool)
            self._total_coverage = nr_rows
        else:
            self._all_data = selection
            if self._binary_target is not None:
                self._binary_target = self._binary_target & selection
            self._total_coverage = int(np.count_nonzero(selection))

    @staticmethod
    def with_comparator(cmp):
        return SubgroupSet(comparator=cmp)

    def _empty_copy(self):
        """'like this one but empty' - copies the ROCList too (see the java FIXME)"""
        s = SubgroupSet(self._maximum_size if self._maximum_size != INT_MAX else 0, None, self._nr_rows,
                        self._binary_target, self._nominal_target_setting)
        s._roc_list = self._roc_list
        s._total_coverage = self._total_coverage
        return s

    def add(self, subgroup):
        """NOTE do not rely on the return value (as in java)"""
        if subgroup is None:
            return False
        if subgroup.get_measure_value() < self._lowest_score:
            return False
        TreeSet.add(self, subgroup)
        self._update()
        return True

    def has_potential(self, quality):
        """includes equal scores, they might be ordered before the old one"""
        return math.isnan(self._lowest_score) or quality >= self._lowest_score

    def _update(self):
        while self._maximum_size < TreeSet.__len__(self):
            self.poll_last()
        if self._maximum_size == TreeSet.__len__(self):
            self._lowest_score = self.last().get_measure_value()

    def get_nr_rows(self):
        """rows in the table, whatever the selection"""
        return self._nr_rows

    def get_total_coverage(self):
        """rows used for mining (all, or the selection)"""
        return self._total_coverage

    def get_total_target_coverage(self):
        return int(np.count_nonzero(self._binary_target))

    def get_all_data_bit_set_clone(self):
        return self._all_data.copy()

    def get_best_score(self):
        return float('nan') if self.is_empty() else self.first().get_measure_value()

    def set_binary_target(self, binary_target):
        """destructive, always reset it afterwards (CAUC)"""
        self._binary_target = binary_target

    def clear(self):
        TreeSet.clear(self)
        if self._roc_list is not None:
            self._roc_list.clear()
        self._lowest_score = float('nan')
        self._joint_entropy = float('nan')

    def size(self):
        return TreeSet.__len__(self)

    # ------------------------------------------------------------------
    # post-processing
    # ------------------------------------------------------------------

    def post_process(self, search_strategy):
        """only does something for COVER_BASED_BEAM_SELECTION"""
        if search_strategy != SearchStrategy.COVER_BASED_BEAM_SELECTION:
            return self
        from . import cover_based_subgroup_selection as cbss
        copy = self._empty_copy()
        for c in cbss.post_process_result_set(self, COVER_BASED_SUBGROUP_SELECTION_TOP_K):
            copy.add(c.get_subgroup())
        copy._update()
        copy._lowest_score = copy.last().get_measure_value()
        return copy

    def set_ids(self):
        for i, s in enumerate(self, 1):
            s.set_id(i)

    def post_process_get_cover_redundancy_and_joint_entropy(self, top_k):
        """only the joint entropy is returned, the cover redundancy is logged"""
        if top_k <= 0:
            raise ValueError("SubgroupSet.postProcessGetCoverRedundancyAndJointEntropy() invalid topK: " + str(top_k))
        n_sg = min(top_k, len(self))
        if n_sg == 0:
            return 0.0
        nr_rows = self._nr_rows
        rows = np.zeros((nr_rows, n_sg), dtype=bool)
        for idx, s in enumerate(self):
            if idx == n_sg:
                break
            b = s.get_members()
            rows[:b.shape[0], idx] |= b[:nr_rows]
        cover_counts = rows.sum(axis=1)
        cover_sum = int(cover_counts.sum())
        total = float(nr_rows)
        expected = cover_sum / total
        cr = 0.0
        for c in cover_counts:
            cr += abs(int(c) - expected)
        cr = div(div(cr, expected), total)

        # group identical rows, in the order the java comparator sorts them:
        # fewer subgroups first, then rows covered by the first subgroup first
        uniq, counts = np.unique(rows, axis=0, return_counts=True)
        keys = [~uniq[:, k] for k in range(n_sg - 1, -1, -1)] + [uniq.sum(axis=1)]
        order = np.lexsort(keys)
        entropy = 0.0
        for k in order:
            frac = counts[k] / total
            entropy += -frac * math.log(frac)
        entropy /= math.log(2.0)
        from .java_compat import format_fixed
        log.log_command_line("CCSUM=%d\tN=%d\tCCEXPECTED=%s\tCR(%d)=%s\tH(%d)=%s" % (
            cover_sum, nr_rows, format_fixed(expected), top_k, format_fixed(cr), top_k, format_fixed(entropy)))
        return entropy

    def get_pattern_team(self, table, k):
        from .binary_table import BinaryTable
        self._binary_table = BinaryTable.from_subgroup_set(table, self)
        subset = self._binary_table.get_approximate_miki(k)
        result = self._empty_copy()
        for index, s in enumerate(self):
            if subset.get(index):
                result.add(s)
        result._joint_entropy = subset.get_joint_entropy()
        result._update()
        return result

    def get_grouping(self, pattern_team):
        grouping = [self._empty_copy() for _ in pattern_team]
        if self._binary_table is None:
            return grouping   # get_pattern_team() was not called first
        team = list(pattern_team)
        for i, s in enumerate(self):
            closest_id = -1
            best = float('-inf')
            closest = None
            for j, p in enumerate(team):
                corr = abs(self._binary_table.compute_correlation(i, j))
                if closest is None or corr > best:
                    closest = p
                    closest_id = j
                    best = corr
            grouping[closest_id].add(s)
        return grouping

    def get_joint_entropy(self):
        return self._joint_entropy

    def filter_subgroups(self, minimum_improvement):
        """Drops a Subgroup when it specialises another one and is not better
        (by more than minimum_improvement). Quadratic in the result size."""
        size = len(self)
        items = self.to_list()
        present = list(items)
        removed = set()
        for first in items:
            for second in present:
                if second is first or id(second) in removed:
                    continue
                if first.strictly_specialises(second) and first.get_measure_value() <= second.get_measure_value() + minimum_improvement:
                    removed.add(id(first))
                    break
            present = [s for s in present if id(s) not in removed]
        if removed:
            self.remove_if(lambda s: id(s) in removed)
        print("Raw result set size: " + str(size))
        print("Filtered result set size: " + str(len(self)))

    def save_extent(self, writer, table, subset, target_concept):
        log.log_command_line("saving extent...")
        try:
            members = [s.get_members() for s in self]
            test_row = "test " + ",0" * len(self) + "\n"
            k = 0
            for i in range(table.get_nr_rows()):
                # cross-validation columns are shorter than the originals, pad
                if subset[i]:
                    row = "train" + "".join(",1" if b[k] else ",0" for b in members) + "\n"
                    writer.write(row)
                    k += 1
                else:
                    writer.write(test_row)
        except OSError as e:
            log.log_command_line("SubgroupSet.saveExtent(): error on file: " + str(e))

    # ------------------------------------------------------------------
    # ROC
    # ------------------------------------------------------------------

    def get_binary_target_clone(self):
        if self._binary_target is None:
            return None
        return self._binary_target.copy()

    def get_roc_list(self):
        if not self._nominal_target_setting or self._binary_target is None:
            return None
        from .roc_list import ROCList
        self._roc_list = ROCList(self)
        return self._roc_list

    def get_roc_list_subgroups(self):
        """rows of (ID, FPR, TPR, Conditions) for the ROC window"""
        out = []
        items = self.to_list()
        for p in self._roc_list:
            s = None
            for s in items:
                if s.get_id() >= p.ID:
                    break
            out.append((s.get_id(), p.get_fpr(), p.get_tpr(), str(s.get_conditions())))
        return out

    def get_roc_list_subgroup_set(self):
        result = self._empty_copy()
        items = self.to_list()
        for p in self._roc_list:
            s = None
            for s in items:
                if s.get_id() >= p.ID:
                    break
            result.add(s)
        return result

    def mark_alternative_descriptions(self):
        if not SubgroupSet.CHECK_FOR_ALTERNATIVE_DESCRIPTIONS:
            return
        if len(self) <= 1:
            return
        from . import condition_list_builder as clb
        from .subgroup import Subgroup
        same = []
        last = Subgroup(clb.empty_list(), self.get_all_data_bit_set_clone(), self)
        last.set_measure_value(float('nan'))
        for s in self:
            s.set_p_value(float('nan'))
            has_same = (last.get_coverage() == s.get_coverage() and last.get_measure_value() == s.get_measure_value()
                        and last.get_secondary_statistic() == s.get_secondary_statistic()
                        and last.get_tertiary_statistic() == s.get_tertiary_statistic())
            if not has_same and not same:
                last = s
                continue
            elif not has_same and same:
                _mark_alternative_descriptions_process(same)
                same = []
                last = s
                continue
            elif has_same and not same:
                same.append(last)
                same.append(s)
            else:
                same.append(s)

    def print(self):
        for s in self:
            log.log_command_line("%d,%d,%s" % (s.get_id(), s.get_coverage(), double_to_string(s.get_measure_value())))


def _mark_alternative_descriptions_process(same):
    from . import condition_list_builder as clb
    n = len(same)
    for i in range(n - 1):
        si = same[i]
        if not math.isnan(si.get_p_value()):
            continue
        sid = si.get_id()
        for k in range(i + 1, n):
            sk = same[k]
            if not math.isnan(sk.get_p_value()):
                continue
            if not np.array_equal(si.get_members(), sk.get_members()):
                continue
            if clb.is_canonicalised_proper_subset_of(si.get_conditions(), sk.get_conditions()):
                if sid > 0:
                    sid = -sid
            si.set_p_value(sid)
            sk.set_p_value(sid)
