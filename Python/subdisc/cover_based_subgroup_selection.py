"""Cover-based subgroup selection (van Leeuwen & Knobbe, ECML PKDD 2011),
with the corrected measure so negative qualities behave."""

import numpy as np

from . import log
from .candidate import Candidate
from .tree_set import TreeSet

USE_CORRECTED_MEASURE = True
DEBUG_PRINTS = False
ALPHA = 0.9


def post_process_candidate_set(candidates, top_k):
    return _post_process(candidates, top_k, True)


def post_process_result_set(subgroups, top_k):
    # lazy, do not write the whole method twice
    cs = TreeSet()
    for s in subgroups:
        cs.add(Candidate(s))
    return _post_process(cs, top_k, False)


def _post_process(candidates, top_k, is_for_candidate_set):
    size = len(candidates)
    log.log_command_line("cbss temporary queue: " + str(size))
    if size == 0:
        return TreeSet()
    if size <= top_k:
        return candidates

    used = np.zeros(size, dtype=bool)
    cand = candidates.to_list()
    last_scores = [c.get_priority() for c in cand]
    members = []
    for c in cand:
        s = c.get_subgroup()
        members.append(s.get_members())
        if is_for_candidate_set:
            s.kill_members()

    nr_rows = cand[0].get_subgroup().get_parent_set().get_total_coverage()
    # members masks are as long as the table, the cover counts too
    cover_counts = np.zeros(max(nr_rows, members[0].shape[0]), dtype=np.int64)
    result = TreeSet()

    _print("loop 0")
    _update(is_for_candidate_set, used, cand, members, cover_counts, result, 0, 1.0, last_scores[0])

    unset = -1
    for i in range(1, min(size, top_k)):
        _print("loop " + str(i))
        best_index = unset
        max_score = float('nan')
        best_weight = float('nan')
        for j in np.flatnonzero(~used[1:]) + 1:
            j = int(j)
            # score can not get better than last time (do not use <=)
            if best_index != unset and last_scores[j] < max_score:
                continue
            c = cand[j]
            priority = c.get_priority()
            weight = _multiplicative_weight(members[j], c.get_subgroup().get_coverage(), cover_counts)
            score = (priority - ((1.0 - weight) * priority)) if USE_CORRECTED_MEASURE else (weight * priority)
            last_scores[j] = score
            # > picks the first of equal scoring candidates
            if best_index == unset or score > max_score:
                best_index = j
                max_score = score
                best_weight = weight
        _update(is_for_candidate_set, used, cand, members, cover_counts, result, best_index, best_weight, max_score)

    if DEBUG_PRINTS:
        _print("used: %s\n" % list(np.flatnonzero(used)))
    return result


def _update(is_for_candidate_set, used, cand, members, cover_counts, result, best_index, best_weight, max_score):
    used[best_index] = True
    c = cand[best_index]
    _print(str(c.get_subgroup()))
    _print("best (%d): %r, %r, %r\n" % (best_index, c.get_priority(), best_weight, max_score))
    if is_for_candidate_set:
        c.set_priority(max_score)
    result.add(c)
    m = members[best_index]
    cover_counts[:m.shape[0]][m] += 1


def _multiplicative_weight(m, coverage, cover_counts):
    # sum of ALPHA^count over the members, in row order like the java loop
    counts = cover_counts[:m.shape[0]][m]
    total = 0.0
    for v in np.power(ALPHA, counts.astype(np.float64)):
        total += float(v)
    return total / coverage if coverage else float('nan')


def _print(msg):
    if DEBUG_PRINTS:
        log.log_command_line(msg)
