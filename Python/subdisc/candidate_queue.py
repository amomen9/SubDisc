"""Candidates waiting to be refined.

Beam strategies keep a current level and a next level; when the current
level is empty the next one becomes current (moveToNextLevel). ROC_BEAM
keeps the next level as a ROC convex hull, cover-based beam selection
selects the next level from a temporary queue.
"""

from . import log
from .candidate import Candidate
from .search_strategy import SearchStrategy
from .tree_set import TreeSet

USE_OLD_COVER_BASED_SUBGROUP_SELECTION = False
INT_MAX = 2147483647


class CandidateQueue:
    def __init__(self, search_parameters, root_candidate):
        if search_parameters is None or search_parameters.get_search_strategy() is None or root_candidate is None:
            raise ValueError("arguments can not be null")
        self._search_strategy = search_parameters.get_search_strategy()
        self._is_beam = self._search_strategy.is_beam()
        self._max_depth = search_parameters.get_search_depth()
        if self._max_depth <= 0:
            raise ValueError("search depth must be > 0")
        # TODO MM (java) check (itsMaximumQueueSize > 0)
        if self._search_strategy.requires_search_width_parameter():
            self._maximum_queue_size = search_parameters.get_search_strategy_width()
        else:
            self._maximum_queue_size = INT_MAX

        self._next_queue = None
        self._next_queue_hull = None
        self._temp_queue = None
        ss = self._search_strategy
        if ss == SearchStrategy.BEAM:
            self._queue = TreeSet()
            self._next_queue = TreeSet()
        elif ss == SearchStrategy.ROC_BEAM:
            from .convex_hull_roc_naive import ConvexHullROCNaive
            self._queue = TreeSet()
            self._next_queue_hull = ConvexHullROCNaive()
        elif ss == SearchStrategy.COVER_BASED_BEAM_SELECTION:
            self._queue = TreeSet()
            self._next_queue = TreeSet()
            self._temp_queue = TreeSet()
        elif ss == SearchStrategy.BEST_FIRST:
            self._queue = TreeSet()
        elif ss in (SearchStrategy.DEPTH_FIRST, SearchStrategy.BREADTH_FIRST):
            self._queue = TreeSet(Candidate.get_comparator(ss))
        else:
            raise ValueError(str(ss))
        self._queue.add(root_candidate)

    def add(self, candidate):
        """add to the right level, the queue is trimmed to the beam width"""
        sg = candidate.get_subgroup()
        if sg.get_depth() >= self._max_depth:
            return False
        # no useful refinement can come from this
        if sg.get_coverage() <= 1:
            return False
        ss = self._search_strategy
        # ROC_BEAM kills the members after the ROC point is made
        if ss != SearchStrategy.ROC_BEAM:
            sg.kill_members()
        if ss == SearchStrategy.BEAM:
            return self._add_to_queue(self._next_queue, candidate)
        if ss == SearchStrategy.ROC_BEAM:
            from .roc_point import SubgroupROCPoint
            added = self._next_queue_hull.add(SubgroupROCPoint(sg))
            sg.kill_members()
            return added
        if ss == SearchStrategy.COVER_BASED_BEAM_SELECTION:
            # just add, whatever the size of the temp queue
            return self._temp_queue.add(candidate)
        return self._add_to_queue(self._queue, candidate)

    def _add_to_queue(self, queue, candidate):
        added = queue.add(candidate)
        if added and len(queue) > self._maximum_queue_size:
            queue.poll_last()
        return added

    def remove_first(self):
        """head of the queue, moves to the next level when needed"""
        if self._is_beam and len(self._queue) == 0:
            self._move_to_next_level()
        return self._queue.poll_first()

    def _move_to_next_level(self):
        log.log_command_line("\nLevel finished --------------------------------------------\n")
        ss = self._search_strategy
        if ss == SearchStrategy.BEAM:
            self._queue = self._next_queue
            self._next_queue = TreeSet()
        elif ss == SearchStrategy.ROC_BEAM:
            from .convex_hull_roc_naive import ConvexHullROCNaive
            from . import process
            if process.ROC_BEAM_TEST:
                print("ROC_BEAM for next level:")
                print("PRE (0.0, TPR) removal:")
                self._next_queue_hull.debug()
                self._next_queue_hull.remove_pure_subgroups()
                print("POST (0.0, TPR) removal:")
                self._next_queue_hull.debug()
            self._queue = self._next_queue_hull.to_tree_set()
            self._next_queue_hull = ConvexHullROCNaive()
        elif ss == SearchStrategy.COVER_BASED_BEAM_SELECTION:
            from . import cover_based_subgroup_selection as cbss
            self._queue = cbss.post_process_candidate_set(self._temp_queue, self._maximum_queue_size)
            self._next_queue = TreeSet()
            self._temp_queue = TreeSet()
        else:
            raise AssertionError(str(ss))

    def size(self):
        """total number of Candidates (current plus next level)"""
        ss = self._search_strategy
        if ss == SearchStrategy.BEAM:
            return len(self._queue) + len(self._next_queue)
        if ss == SearchStrategy.ROC_BEAM:
            return len(self._queue) + self._next_queue_hull.size()
        if ss == SearchStrategy.COVER_BASED_BEAM_SELECTION:
            return len(self._queue) + len(self._temp_queue)
        return len(self._queue)

    def current_level_queue_size(self):
        return len(self._queue)
