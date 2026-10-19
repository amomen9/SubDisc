import pytest

from conftest import load
from subdisc import condition_list_builder as clb
from subdisc.condition import Condition
from subdisc.condition_base import ConditionBase
from subdisc.operators import Operator


@pytest.fixture(scope="module")
def table():
    return load("adult.txt")


def cl(column, op, value, previous=None):
    cb = ConditionBase(column, op)
    c = Condition(cb, value) if isinstance(value, str) else Condition(cb, float(value), 0)
    return clb.create_list(c) if previous is None else clb.create_list(previous, c)


def test_1(table):
    # (workclass = 'Private') and (education = 'Bachelors') do not specialise each other
    a = cl(table.get_columns()[1], Operator.EQUALS, "Private")
    b = cl(table.get_columns()[3], Operator.EQUALS, "Bachelors")
    assert a.size() == 1 and b.size() == 1
    assert not a.strictly_specialises(b)
    assert not b.strictly_specialises(a)


def test_2(table):
    # (age <= 20) specialises (age <= 30), not the other way round
    a = cl(table.get_columns()[0], Operator.LESS_THAN_OR_EQUAL, 20)
    b = cl(table.get_columns()[0], Operator.LESS_THAN_OR_EQUAL, 30)
    assert a.strictly_specialises(b)
    assert not b.strictly_specialises(a)


def test_3(table):
    # (age <= 20) does not specialise (education-num <= 8)
    a = cl(table.get_columns()[0], Operator.LESS_THAN_OR_EQUAL, 20)
    b = cl(table.get_columns()[4], Operator.LESS_THAN_OR_EQUAL, 8)
    assert not a.strictly_specialises(b)
    assert not b.strictly_specialises(a)


def test_4(table):
    # (age >= 40) specialises (age >= 30), not the other way round
    a = cl(table.get_columns()[0], Operator.GREATER_THAN_OR_EQUAL, 40)
    b = cl(table.get_columns()[0], Operator.GREATER_THAN_OR_EQUAL, 30)
    assert a.strictly_specialises(b)
    assert not b.strictly_specialises(a)


def test_5(table):
    age, work = table.get_columns()[0], table.get_columns()[1]
    geq, eq = Operator.GREATER_THAN_OR_EQUAL, Operator.EQUALS
    y = cl(age, geq, 40)
    # (age >= 40 ^ workclass = 'Private') specialises (age >= 40)
    x = cl(work, eq, "Private", cl(age, geq, 40))
    assert x.size() == 2
    assert x.strictly_specialises(y)
    assert not y.strictly_specialises(x)
    # the other order, same result
    x = cl(age, geq, 40, cl(work, eq, "Private"))
    assert x.strictly_specialises(y)
    assert not y.strictly_specialises(x)
    # numeric threshold the wrong way
    x = cl(work, eq, "Private", cl(age, geq, 30))
    assert not x.strictly_specialises(y)
    assert not y.strictly_specialises(x)
    # and the right way
    x = cl(work, eq, "Private", cl(age, geq, 50))
    assert x.strictly_specialises(y)
    assert not y.strictly_specialises(x)
