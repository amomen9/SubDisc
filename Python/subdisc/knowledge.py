"""External knowledge for the propensity score measures: the .gkf (global)
and .lkf (local) files with one ConditionList per line, read from the
current directory (java LocalKnowledge, GlobalKnowledge and
ExternalKnowledgeFileLoader)."""

import os

import numpy as np

from . import log
from . import condition_list_builder as clb
from .attribute_type import AttributeType
from .condition import Condition, UNINITIALISED_SORT_INDEX
from .condition_base import ConditionBase
from .java_compat import parse_float
from .operators import Operator
from .propensity_score import StatisticsBayesRule

EXTERNAL_KNOWLEDGE_STANDARD_DIR = ""
OPERATORS = sorted({" %s " % o.GUI_TEXT for o in Operator})


class LocalKnowledge:
    def __init__(self, target, explanatory_conditions):
        self._column_to_lists = {}
        self._list_to_bits = {}
        self._list_to_rule = {}
        if not explanatory_conditions:
            return
        for cl in explanatory_conditions:
            for c in cl:
                col = c.get_column()
                log.log_command_line("Local Knowledge variables")
                log.log_command_line(col.get_name())
                self._column_to_lists.setdefault(id(col), (col, []))[1].append(cl)
        n = explanatory_conditions[0].get(0).get_column().size()
        all_rows = np.ones(n, dtype=bool)
        tc = int(np.count_nonzero(target))
        for cl in explanatory_conditions:
            bs = all_rows.copy()
            for c in cl:
                b = c.get_column().evaluate(all_rows, c)
                bs &= b
                _print("condition", int(np.count_nonzero(b)))
            self._list_to_bits[id(cl)] = bs
            _print("target", tc)
            _print("known subgroup", int(np.count_nonzero(bs)))
            self._list_to_rule[id(cl)] = StatisticsBayesRule(bs, target)

    def _lists(self, conditions, is_bayes):
        out = []
        seen = set()
        for c in conditions:
            entry = self._column_to_lists.get(id(c.get_column()))
            if entry is not None:
                for cl in entry[1]:
                    if id(cl) not in seen:
                        seen.add(id(cl))
                        out.append(cl)
            if is_bayes:
                log.log_command_line(c.get_column().get_name())
        return out

    def get_bit_sets(self, conditions):
        return [self._list_to_bits[id(cl)] for cl in self._lists(conditions, False)]

    def get_statistics_bayes_rule(self, conditions):
        return [self._list_to_rule[id(cl)] for cl in self._lists(conditions, True)]


class GlobalKnowledge:
    def __init__(self, target, explanatory_conditions):
        self._bit_sets = []
        self._rules = []
        if not explanatory_conditions:
            return
        n = explanatory_conditions[0].get(0).get_column().size()
        all_rows = np.ones(n, dtype=bool)
        for cl in explanatory_conditions:
            bs = all_rows.copy()
            for c in cl:
                bs = c.get_column().evaluate(bs, c)
            self._bit_sets.append(bs)
            self._rules.append(StatisticsBayesRule(bs, target))

    def get_bit_sets2(self):
        return list(self._bit_sets)

    def get_statistics_bayes_rule2(self):
        return list(self._rules)


class ExternalKnowledgeFileLoader:
    def __init__(self, table, condition_bases):
        g = _lines_from_file(".gkf")
        l = _lines_from_file(".lkf")
        _print_lines(True, g)
        _print_lines(False, l)
        self._global = _knowledge_to_conditions(table, condition_bases, g)
        self._local = _knowledge_to_conditions(table, condition_bases, l)

    def get_knowledge(self, is_global):
        return self._global if is_global else self._local


def _lines_from_file(ext):
    d = os.path.abspath(EXTERNAL_KNOWLEDGE_STANDARD_DIR or ".")
    files = sorted(f for f in os.listdir(d) if f.lower().endswith(ext))
    # java crashes when there is no file (the empty list was never returned)
    if not files:
        return []
    out = []
    try:
        with open(os.path.join(d, files[0]), encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if line == "":
                    break
                out.append(line)
    except OSError as e:
        log.log_command_line(str(e))
    return out


def _print_lines(is_global, lines):
    log.log_command_line("\nGlobal External Knowledge:" if is_global else "Local External Knowledge:")
    for s in lines:
        log.log_command_line(s)
    log.log_command_line("")


def _knowledge_to_conditions(table, condition_bases, knowledge):
    if not knowledge:
        return []
    cbl = condition_bases.get_condition_bases()
    out = []
    for line in knowledge:
        cl = clb.empty_list()
        for s in line.split(" AND "):
            name, op_text, value = _disect(s)
            col = table.get_column(name)
            op = Operator.from_string(op_text)
            cb = next((c for c in cbl if c.get_column() is col and c.get_operator() == op), None)
            # NOTE java swaps the two cases here and then crashes on null
            b = cb if cb is not None else ConditionBase(col, op)
            t = col.get_type()
            if t == AttributeType.NOMINAL:
                cond = Condition(b, value)
            elif t == AttributeType.NUMERIC:
                cond = Condition(b, parse_float(value), UNINITIALISED_SORT_INDEX)
            elif t == AttributeType.BINARY:
                if not AttributeType.is_valid_binary_value(value):
                    raise ValueError(value + " is not a valid BINARY value")
                cond = Condition(b, AttributeType.is_valid_binary_true_value(value))
            else:
                raise AssertionError(t)
            cl = clb.create_list(cl, cond)
            if cb is None:
                log.log_command_line("Irrelevant knowledge: '%s' in '%s'" % (cond, line))
        out.append(cl)
        log.log_command_line("ADDED=" + str(cl))
    return out


def _disect(cond):
    for s in OPERATORS:
        if s in cond:
            tmp = cond.split(s)
            if tmp[1].startswith("'") and tmp[1].endswith("'"):
                tmp[1] = tmp[1][1:-1]
            return tmp[0], s.strip(), tmp[1]
    raise ValueError("ExternalKnowledgeFileLoader can not parse: " + cond)


def _print(subject, cardinality):
    log.log_command_line("cardinality %s\n%d" % (subject, cardinality))
