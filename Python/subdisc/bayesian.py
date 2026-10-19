"""Greedy hill climbing of a Bayesian network (BDeu score) over binary targets.

The random generator is static in java and re-seeded with the clock for
every new Bayesian, so multi-label results are not repeatable there.
Set FIXED_SEED (e.g. 12345, the value commented out in java) to get the
same networks every run.
"""

import time

from .dag import DAG
from .java_compat import JavaRandom

FIXED_SEED = None


class _EdgeQuality:
    __slots__ = ("qual_one", "qual_two", "node1", "node2")

    def __init__(self):
        self.qual_one = -100000.0
        self.qual_two = -100000.0
        self.node1 = 0
        self.node2 = 0

    def set_edge(self, i, j):
        self.node1 = i
        self.node2 = j

    def set_quality(self, x, y):
        self.qual_one = x
        self.qual_two = y

    def get_quality(self):
        return self.qual_two

    def get_action(self):
        return self.qual_one


class Bayesian:
    _random = None   # static in java

    def __init__(self, table, targets=None):
        self._dag = DAG(targets if targets is not None else table.get_nr_columns())
        self._table = table
        seed = FIXED_SEED if FIXED_SEED is not None else int(time.time() * 1000)
        Bayesian._random = JavaRandom(seed)

    def get_dag(self):
        return self._dag

    def test(self):
        self.climb()

    def _compute_quality(self, child):
        t = self._table.select_columns(self._dag.get_node(child).get_parents())
        t.add_column(self._table.get_column(child))   # the child is the last column
        return t.compute_bdeu_faster()

    def score_next(self):
        dag = self._dag
        q = _EdgeQuality()
        best = -100000.0
        size = dag.get_size()
        for i0 in range(size):
            for j0 in range(i0):
                i, j = i0, j0
                c = dag.get_node(j).is_connected(i)
                if c == 0:
                    # i -/- j
                    if dag.add_arc_acyclic(i, j, False):
                        qual = self._compute_quality(j)
                        if (qual - dag.get_quality(j)) > best:
                            best = qual - dag.get_quality(j)
                            q.set_quality(1, qual)
                            q.set_edge(i, j)
                        dag.remove_arc(i, j, False)
                    else:
                        dag.add_arc(j, i, False)
                        qual = self._compute_quality(i)
                        if (qual - dag.get_quality(i)) > best:
                            best = qual - dag.get_quality(i)
                            q.set_quality(1, qual)
                            q.set_edge(j, i)
                        dag.remove_arc(j, i, False)
                        continue
                    if dag.add_arc_acyclic(j, i, False):
                        qual = self._compute_quality(i)
                        if (qual - dag.get_quality(i)) > best:
                            best = qual - dag.get_quality(i)
                            q.set_quality(1, qual)
                            q.set_edge(j, i)
                        dag.remove_arc(j, i, False)
                    continue
                if c == 1:
                    # i --> j, swap and fall through
                    i, j = j, i
                # i <-- j
                dag.remove_arc(j, i, False)
                qual = self._compute_quality(i)
                qual_tmp = qual
                if (qual - dag.get_quality(i)) > best:
                    best = qual - dag.get_quality(i)
                    q.set_quality(0, qual)
                    q.set_edge(j, i)
                if dag.add_arc_acyclic(i, j, False):
                    # only reverse non-covered arcs
                    covered = False
                    if dag.get_node(j).get_nr_parents() - 1 == dag.get_node(i).get_nr_parents():
                        diff = dag.get_node(i).get_parents().symmetric_difference(dag.get_node(j).get_parents())
                        if diff.get_item_count() == 1 and diff.get(i):
                            covered = True
                    if not covered:
                        qual = self._compute_quality(j)
                        if (qual - dag.get_quality(j)) + (qual_tmp - dag.get_quality(i)) > best:
                            best = (qual - dag.get_quality(j)) + (qual_tmp - dag.get_quality(i))
                            q.set_quality(qual_tmp, qual)
                            q.set_edge(i, j)
                    dag.remove_arc(i, j, False)
                dag.add_arc(j, i, False)
        return q

    def climb(self):
        dag = self._dag
        model_q = 0.0
        loc_max = False
        trials = 0
        escape = 0
        for i in range(dag.get_size()):
            model_q += dag.get_quality(i)
        # 2 * size works for larger graphs, more is better if you have the time
        cycle = 2 * dag.get_size()
        while not loc_max and cycle > 0:
            cycle -= 1
            model_q += self.rcar(7)
            best = self.score_next()
            tmp = model_q
            tmp -= dag.get_quality(best.node2)
            tmp += best.get_quality()
            if best.get_action() < 0:   # reverse: also for node 1
                tmp -= dag.get_quality(best.node1)
                tmp += best.get_quality()
            loc_max = tmp < model_q
            if not loc_max:
                trials = 0
                dag.set_quality(best.node2, best.get_quality())
                action = int(best.get_action())
                if action == 0:
                    dag.remove_arc(best.node1, best.node2, True)
                elif action == 1:
                    dag.add_arc(best.node1, best.node2, True)
                else:
                    dag.set_quality(best.node1, best.get_action())
                    dag.remove_arc(best.node2, best.node1, True)
                    dag.add_arc(best.node1, best.node2, True)
                model_q = tmp
            elif trials < 4:   # MAXTRIAL
                if escape > 12:   # MAXESCAPE
                    return
                model_q += self.rcar(7)
                loc_max = False
                trials += 1
                escape += 1

    def rcar(self, r):
        dag = self._dag
        rnd = Bayesian._random
        check = set()
        cov_edges = []
        rr = int(rnd.next_float() * r)
        for _ in range(rr):
            for i in range(dag.get_size()):
                child = dag.get_node(i)
                parents = child.get_parents()
                for j in range(dag.get_size()):
                    if parents.get(j):
                        parent = dag.get_node(j)
                        if child.get_nr_parents() - 1 == parent.get_nr_parents():
                            diff = child.get_parents().symmetric_difference(parent.get_parents())
                            if diff.get_item_count() == 1 and diff.get(j):
                                e = _EdgeQuality()
                                e.set_edge(j, i)
                                cov_edges.append(e)
            if cov_edges:
                # sic: the list is never emptied in java (unlike the C++ original)
                e = int(rnd.next_float() * len(cov_edges))
                edge = cov_edges[e]
                dag.remove_arc(edge.node1, edge.node2, False)
                dag.add_arc(edge.node2, edge.node1, False)
                check.add(edge.node1)
                check.add(edge.node2)
        score_diff = 0.0
        for nd in sorted(check):
            score_diff -= dag.get_quality(nd)
            score_diff += self._compute_quality(nd)
        return score_diff
