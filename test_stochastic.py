from pytest import approx
from stochastic import Stochastic


def test_d6():
    v = Stochastic.uniform(range(1, 7))
    assert v.expected() == approx(3.5)
    assert v.stddev() == approx(1.71, rel=1e-2)


def test_2d6():
    v = Stochastic.uniform(range(1, 7)) * 2
    assert v.expected() == 7
    assert v.stddev() == approx(2.41, rel=1e-2)
