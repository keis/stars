import operator
from collections import deque
from functools import reduce
from itertools import islice
from math import factorial

import pytest  # type: ignore
from pytest import approx  # type: ignore
from hypothesis import given
from hypothesis.strategies import composite, integers, lists
from stochastic import Stochastic, apply


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


def comb(n, k):
    return factorial(n + k - 1) / (factorial(k) * factorial(n - 1))


def d6():
    return Stochastic.uniform(range(1, 7))


@given(integers().flatmap(addends))
def test_sum(input):
    total, addends = input
    assert sum(addends) == total


def test_d6():
    v = d6()
    assert v.expected() == approx(3.5)
    assert v.stddev() == approx(1.71, rel=1e-2)


def test_2d6():
    v = d6() * 2
    assert v.expected() == 7
    assert v.stddev() == approx(2.41, rel=1e-2)


def test_2d6_bag():
    v = apply([d6().bag(2)], lambda v: v[0] + v[1])
    assert v.expected() == approx(7)
    assert v.stddev() == approx(2.41, rel=1e-2)


@pytest.mark.parametrize('n', [2, 4, 6, 8])
def test_nd6_mul(n, benchmark):
    benchmark(lambda: d6() * n)


@pytest.mark.parametrize('n', [2, 4, 6, 8])
def test_nd6_apply(n, benchmark):
    benchmark(lambda: apply([d6()] * n, lambda *s: sum(s)))


@pytest.mark.parametrize('n', [2, 4, 6, 8])
def test_nd6_bag(n, benchmark):
    benchmark(lambda: apply([d6().bag(n)], sum))


@pytest.mark.parametrize('k', [2, 4, 6, 8])
def test_uniform_bag(k, benchmark):
    v = d6()
    c = benchmark(lambda: v.bag(k))
    assert len(c) == comb(6, k)


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
