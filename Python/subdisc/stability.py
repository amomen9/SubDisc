"""Subsampling stability of a mining result (not in the java version).

The same search is run B times, each time on a random subsample of the
rows (without replacement, 80% by default). A subgroup that is real should
come back in most runs; one that only shows up because of a few lucky rows
comes and goes.

Per original subgroup S:
    exact     share of runs with a subgroup that has the very same description
    cover     share of runs with a subgroup whose rows (on the full data)
              have a Jaccard similarity >= tau with the rows of S, where
              J(A, B) = |A and B| / |A or B|
    mean J    the best Jaccard per run, averaged over the runs
The 'cover' number is the useful one for numeric conditions: in a
subsample 'age >= 33' may become 'age >= 34', which is the same subgroup in
all but name.

Why no bootstrap (drawing rows with replacement)? Duplicated rows would
change coverage counts and the minimum coverage, subsampling keeps every
row at most once and needs no copy of the table.
"""

import copy

import numpy as np

from . import log


class Canceller:
    """stands in for the main window during the runs: the search asks it
    is_cancelled() and tells it its progress via set_mining_title()"""

    def __init__(self, progress=None):
        self.cancelled = False
        self.progress = progress
        self.prefix = ""

    def is_cancelled(self):
        return self.cancelled

    def set_mining_title(self, text):
        if self.progress is not None:
            self.progress(self.prefix + text)


def jaccard_matrix(a, b):
    """J between every row of a (k x n) and every row of b (l x n), bool arrays"""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]))
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    inter = af @ bf.T
    sa = af.sum(axis=1)[:, None]
    sb = bf.sum(axis=1)[None, :]
    union = sa + sb - inter
    with np.errstate(all="ignore"):
        j = np.where(union > 0, inter / union, 0.0)
    return j


class StabilityRow:
    __slots__ = ("subgroup", "exact", "cover", "mean_jaccard")


def run(table, base_selection, search_parameters, reference, nr_runs=10, fraction=0.8, seed=1,
        threshold=0.8, canceller=None):
    """StabilityRow per reference subgroup, plus a dict of run statistics"""
    from .holdout import evaluate_conditions
    from .process import run_subgroup_discovery
    n = table.get_nr_rows()
    base = np.ones(n, dtype=bool) if base_selection is None else np.asarray(base_selection[:n], dtype=bool)
    pool = np.flatnonzero(base)
    size = int(round(fraction * pool.shape[0]))
    if size < 1:
        raise ValueError("the subsample would be empty")
    reference = list(reference)
    ref_keys = [str(s.get_conditions()) for s in reference]
    ref_covers = np.array([evaluate_conditions(table, s.get_conditions(), base) for s in reference], dtype=bool)
    exact = np.zeros(len(reference))
    cover = np.zeros(len(reference))
    jsum = np.zeros(len(reference))
    rng = np.random.default_rng(seed)
    sizes = []
    found_sets = []
    done = 0
    mem = log.COMMANDLINELOG
    for r in range(nr_runs):
        if canceller is not None and canceller.is_cancelled():
            break
        pick = rng.choice(pool, size=size, replace=False)
        sel = np.zeros(n, dtype=bool)
        sel[pick] = True
        sp = copy.copy(search_parameters)
        sp.set_target_concept(copy.copy(search_parameters.get_target_concept()))
        if canceller is not None:
            canceller.prefix = "run %d of %d: " % (r + 1, nr_runs)
        log.COMMANDLINELOG = False
        try:
            sd = run_subgroup_discovery(table, 0, sel, sp, False, 1, canceller)
        finally:
            log.COMMANDLINELOG = mem
        if canceller is not None and canceller.is_cancelled():
            break   # a stopped run is only part of a result, leave it out
        found = list(sd.get_result()) if sd is not None else []
        sizes.append(len(found))
        keys = set(str(s.get_conditions()) for s in found)
        found_sets.append(keys)
        covers = np.array([evaluate_conditions(table, s.get_conditions(), base) for s in found], dtype=bool)
        if covers.ndim == 1:
            covers = covers.reshape(0, n)
        j = jaccard_matrix(ref_covers, covers)
        best = j.max(axis=1) if j.shape[1] else np.zeros(len(reference))
        exact += np.array([k in keys for k in ref_keys], dtype=float)
        cover += best >= threshold
        jsum += best
        done += 1
    rows = []
    for i, s in enumerate(reference):
        row = StabilityRow()
        row.subgroup = s
        row.exact = exact[i] / done if done else float('nan')
        row.cover = cover[i] / done if done else float('nan')
        row.mean_jaccard = jsum[i] / done if done else float('nan')
        rows.append(row)
    # how much do the runs agree with each other (descriptions)
    pair = []
    for a in range(len(found_sets)):
        for b in range(a + 1, len(found_sets)):
            u = found_sets[a] | found_sets[b]
            if u:
                pair.append(len(found_sets[a] & found_sets[b]) / len(u))
    stats = {
        "runs done": done,
        "runs asked": nr_runs,
        "rows per run": size,
        "mean result size": float(np.mean(sizes)) if sizes else float('nan'),
        "mean description overlap between runs": float(np.mean(pair)) if pair else float('nan'),
        "mean cover stability": float(np.mean(cover / done)) if done and len(reference) else float('nan'),
    }
    return rows, stats
