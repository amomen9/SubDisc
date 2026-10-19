"""The base conditions, built once per search (used for logging and the
external knowledge files)."""

from .attribute_type import AttributeType
from .condition_base import ConditionBase
from .operators import Operator
from .target_type import TargetType
from . import condition_list_builder as clb


class ConditionBaseSet:
    def __init__(self, table, search_parameters):
        self._condition_bases = tuple(_get_base_conditions(table, search_parameters))

    def get_condition_bases(self):
        return self._condition_bases

    def size(self):
        return len(self._condition_bases)

    def __str__(self):
        return "ConditionBaseSet\n" + "".join("\t" + str(c) + "\n" for c in self._condition_bases)

    @staticmethod
    def is_in_condition_base_order(cbs, condition_list):
        """used in asserts only, see the java comment"""
        if condition_list.size() <= 1:
            return True
        lst = cbs._condition_bases
        size = len(lst)
        canonical = clb.to_canonical_order(condition_list)
        nxt = 0
        for i in range(size):
            if nxt >= len(canonical):
                break
            cb = lst[i]
            cmp = cb.get_column().get_index() - canonical[nxt].get_column().get_index()
            if cmp < 0:
                continue
            if cmp == 0:
                cmp2 = cb.get_operator().ordinal - canonical[nxt].get_operator().ordinal
                if cmp2 < 0:
                    continue
                if cmp2 == 0:
                    nxt += 1
                    continue
                return False
            return False
        if len(canonical) <= size:
            return True
        last = lst[size - 1]
        for c in canonical[size:]:
            if c.get_column().get_index() != last.get_column().get_index() or c.get_operator().ordinal != last.get_operator().ordinal:
                return False
        return True


def _get_base_conditions(table, search_parameters):
    tc = search_parameters.get_target_concept()
    is_single_nominal = tc.get_target_type() == TargetType.SINGLE_NOMINAL
    use_sets = search_parameters.get_nominal_sets()
    nom_op = Operator.ELEMENT_OF if (is_single_nominal and use_sets) else Operator.EQUALS
    num_ops = search_parameters.get_numeric_operator_setting().get_operators()
    result = []
    for c in table.get_columns():
        if not c.get_is_enabled() or tc.is_target_attribute(c):
            continue
        t = c.get_type()
        if t == AttributeType.NOMINAL:
            result.append(ConditionBase(c, nom_op))
        elif t == AttributeType.NUMERIC:
            for o in num_ops:
                result.append(ConditionBase(c, o))
        elif t == AttributeType.BINARY:
            result.append(ConditionBase(c, Operator.EQUALS))
        else:
            raise AssertionError(t)
    return result
