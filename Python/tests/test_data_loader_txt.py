import pytest

from conftest import load
from subdisc.attribute_type import AttributeType

N, U, B = AttributeType.NOMINAL, AttributeType.NUMERIC, AttributeType.BINARY


@pytest.mark.parametrize("name", [
    "adult.txt",
    "dataloader no missing.txt",
    "long10k.txt",
    # "long1m.txt",   quite large, meant for scalability testing
    "commas in fields.txt",
    "discretisation.txt",
    "long100k.txt",
    "long10.txt",
    "long with unique nums.txt",
    "missing values.txt",
])
def test_valid_txt(name):
    assert load(name) is not None


def test_size_of_100k():
    t = load("long100k.txt")
    assert t.get_nr_rows() == 100000
    assert t.get_nr_columns() == 19
    types = [B, N, N, N, U, N, U, U, N, N, U, U, N, N, U, U, U, U, U]
    cards = [1, 2, 2, 3, 96, 8, 9000, 60268, 2, 6, 9, 8, 3, 5, 8, 8, 15, 6, 3]
    cols = t.get_columns()
    assert [c.get_type() for c in cols] == types
    assert [c.get_cardinality() for c in cols] == cards
