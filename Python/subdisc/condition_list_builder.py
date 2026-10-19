"""Conjunctions of Conditions.

Java has ConditionList0/1/2/3/N to keep things fast. One class does it
here: conditions are kept in canonical order (for comparing) and in search
order (the order they were added, used for printing and get()). The
strictly_specialises() rules depend on the size, just like the java
classes, and are copied one to one.
"""

from functools import cmp_to_key

from .operators import Operator
from .java_compat import float_compare

CONJUNCT_SIGN = " AND "


def _cond_cmp(a, b):
    return a.compare_to(b)


class ConditionList:
    __slots__ = ("_canonical", "_search")

    def __init__(self, canonical, search):
        self._canonical = canonical
        self._search = search

    def size(self):
        return len(self._search)

    def __len__(self):
        return len(self._search)

    def get(self, index):
        """Conditions in search order."""
        if index < 0:
            raise IndexError(index)
        return self._search[index]

    def get_canonical(self, index):
        if index < 0:
            raise IndexError(index)
        return self._canonical[index]

    def __iter__(self):
        return iter(self._search)

    def compare_to(self, other):
        if self is other:
            return 0
        n = len(self._canonical)
        if n == 0:
            # there is only one canonical empty list, smaller than all others
            return -1
        cmp = n - other.size()
        if cmp != 0:
            return cmp
        for a, b in zip(self._canonical, other._canonical):
            cmp = a.compare_to(b)
            if cmp != 0:
                return cmp
        return 0

    def __lt__(self, other):
        return self.compare_to(other) < 0

    def __str__(self):
        if not self._search:
            return "(empty)"
        return CONJUNCT_SIGN.join(str(c) for c in self._search)

    __repr__ = __str__

    def strictly_specialises(self, other):
        """(A^B) specialises (A), (A^B^C) specialises (A^B), (a > 10) specialises (a > 5)."""
        n = len(self._canonical)
        on = other.size()
        if n == 0:
            return False   # the empty list specialises nothing
        if n >= 4:
            return False   # implement (java)
        if on > n:
            return False
        if on == 0:
            return True
        c = self._canonical
        o = other._canonical
        if n == 1:
            return c[0].strictly_specialises(o[0])
        if n == 2:
            f, s = c
            if on == 1:
                x = o[0]
                if f.logically_equivalent(x) or s.logically_equivalent(x):
                    return True
                if f.strictly_specialises(x) or s.strictly_specialises(x):
                    return True
                return False
            of, os_ = o
            if f.logically_equivalent(of) and s.logically_equivalent(os_):
                return False
            if f.logically_equivalent(os_) and s.logically_equivalent(of):
                return False
            if f.logically_equivalent(of) and s.strictly_specialises(os_):
                return True
            if f.strictly_specialises(of) and s.logically_equivalent(os_):
                return True
            if f.logically_equivalent(os_) and s.strictly_specialises(of):
                return True
            if f.strictly_specialises(os_) and s.logically_equivalent(of):
                return True
            return False
        # n == 3
        f, s, t = c
        if on == 1:
            x = o[0]
            if f.logically_equivalent(x) or s.logically_equivalent(x) or t.logically_equivalent(x):
                return True
            if f.strictly_specialises(x) or s.strictly_specialises(x) or t.strictly_specialises(x):
                return True
            return False
        if on == 2:
            a, b = o
            # is the other list contained in this one?
            for p, q in ((f, s), (f, t), (s, f), (s, t), (t, f), (t, s)):
                if p.logically_equivalent(a) and q.logically_equivalent(b):
                    return True
            return False
        a, b, d = o
        perms = ((f, s, t), (f, t, s), (s, f, t), (s, t, f), (t, f, s), (t, s, f))
        # type 1, 2 and 3: one position strictly specialises, the others equal
        for x, y, z in perms:
            if x.strictly_specialises(a) and y.logically_equivalent(b) and z.logically_equivalent(d):
                return True
        for x, y, z in perms:
            if x.logically_equivalent(a) and y.strictly_specialises(b) and z.logically_equivalent(d):
                return True
        for x, y, z in perms:
            if x.logically_equivalent(a) and y.logically_equivalent(b) and z.strictly_specialises(d):
                return True
        return False


EMPTY_LIST = ConditionList((), ())


def empty_list():
    return EMPTY_LIST


def create_list(condition_list_or_condition, condition=None):
    """create_list(cond) or create_list(condition_list, cond)"""
    if condition is None:
        c = condition_list_or_condition
        if c is None:
            raise ValueError()
        return ConditionList((c,), (c,))
    cl = condition_list_or_condition
    canonical = cl._canonical
    pos = len(canonical)
    for i, e in enumerate(canonical):
        cmp = condition.compare_to(e)
        if cmp == 0:
            # conjunctions of identical Conditions make no sense
            print("ERROR ConditionList%d: duplicate conjuncts\n\t%s\n\t%s" % (len(canonical) + 1, cl, condition))
        if cmp < 0:
            pos = i
            break
    new_canonical = canonical[:pos] + (condition,) + canonical[pos:]
    return ConditionList(new_canonical, cl._search + (condition,))


def to_canonical_order(condition_list):
    return sorted((condition_list.get(i) for i in range(condition_list.size())), key=cmp_to_key(_cond_cmp))


def to_canonical_order_string(condition_list):
    """used for testing results of multi-threaded experiments"""
    return CONJUNCT_SIGN.join(str(c) for c in to_canonical_order(condition_list))


def is_canonicalised_proper_subset_of(former, latter):
    if former.size() >= latter.size():
        return False
    a_former = _canonicalise(former)
    a_latter = _canonicalise(latter)
    for c in a_former:
        # all Conditions in former are checked and occurred in latter
        if c is None:
            return True
        found = False
        for d in a_latter:
            if d is None:
                return False
            if c.compare_to(d) == 0:
                found = True
                break
        if not found:
            return False
    return True


def _canonicalise(condition_list):
    """sorted, with the least specific of two <= (or >=) bounds on the same
    column removed; removed entries end up as None at the end"""
    lst = [condition_list.get(i) for i in range(condition_list.size())]
    n = len(lst)
    if n <= 1:
        return lst
    lst.sort(key=cmp_to_key(_cond_cmp))
    for i in range(n - 1):
        ci = lst[i]
        if ci is None:
            continue
        cc = ci.get_column()
        oi = ci.get_operator()
        if oi != Operator.LESS_THAN_OR_EQUAL and oi != Operator.GREATER_THAN_OR_EQUAL:
            continue
        is_leq = oi == Operator.LESS_THAN_OR_EQUAL
        for j in range(i + 1, n):
            cj = lst[j]
            if cj is None or cc is not cj.get_column() or oi != cj.get_operator():
                continue
            is_leq2 = cj.get_operator() == Operator.LESS_THAN_OR_EQUAL
            if is_leq != is_leq2:
                continue
            assert float_compare(ci.get_numeric_value(), cj.get_numeric_value()) < 0
            # remove the least specific bound
            if is_leq:
                lst[j] = None
            else:
                lst[i] = None
                break
    out = [c for c in lst if c is not None]
    return out + [None] * (n - len(out))
