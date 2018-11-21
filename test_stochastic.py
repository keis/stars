import operator
from collections import deque
from functools import reduce
from itertools import islice
from pytest import approx
from hypothesis import given
from hypothesis.strategies import composite, integers, lists
from stochastic import Stochastic


def window(size, items):
    i = iter(items)
    win = deque(islice(i, size), maxlen=size)
    yield win
    for o in i:
        win.append(o)
        yield win


@composite
def addends(draw, total):
    splits = draw(
        lists(
            integers(min_value=0, max_value=max(total - 1, 0)),
            unique=True,
            max_size=max(total-1, 0),
        ).map(sorted)
    )
    return total, [b - a for a, b in window(2, [0, *splits, total])]


@given(integers().flatmap(addends))
def test_sum(input):
    total, addends = input
    assert sum(addends) == total


def test_d6():
    v = Stochastic.uniform(range(1, 7))
    assert v.expected() == approx(3.5)
    assert v.stddev() == approx(1.71, rel=1e-2)


def test_2d6():
    v = Stochastic.uniform(range(1, 7)) * 2
    assert v.expected() == 7
    assert v.stddev() == approx(2.41, rel=1e-2)


@given(
    integers().flatmap(addends).map(
        lambda v: (Stochastic.constant(v[0]), map(Stochastic.constant, v[1]))
    )
)
def test_equality(input):
    total, addends = input
    result = reduce(operator.add, addends)
    assert result == total
    assert hash(result) == hash(total)
