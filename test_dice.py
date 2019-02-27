import pytest  # type: ignore
from pytest import approx  # type: ignore
from dice import dice


@pytest.mark.parametrize(
    ('spec', 'minv', 'maxv', 'expected'),
    [
        ('d6', 1, 6, 3.5),
        ('d6+1', 2, 7, 4.5),
        ('2d6', 2, 12, 7),
        ('2d6k1', 1, 6, 4.47222),
        ('3d6', 3, 18, 10.5),
        ('4d6k3', 3, 18, 12.2446),
    ]
)
def test_dice(spec, minv, maxv, expected, benchmark):
    d = benchmark(lambda: dice(spec))
    assert min(d) == minv
    assert max(d) == maxv
    assert d.expected() == approx(expected)
