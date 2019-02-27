import operator
from functools import reduce
from math import sqrt, factorial
from typing import (
    Callable, Dict, Union, Iterable, Sequence, Tuple, Type, TypeVar, overload)
from itertools import product, combinations_with_replacement, groupby


T = TypeVar('T')
S = TypeVar('S')
A0 = TypeVar('A0')
A1 = TypeVar('A1')
A2 = TypeVar('A2')

Reduce = Callable[[T, T], T]
StochasticSeq = Iterable[Tuple[T, float]]

def combinations(point) -> float:
    return reduce(
        operator.mul,
        [factorial(len(list(grp))) for _, grp in groupby(point)]
    )

class Stochastic(Dict[T, float]):
    _hash: int

    @classmethod
    def constant(cls: Type['Stochastic[T]'], value: T) -> 'Stochastic[T]':
        return cls(((value, 1),))

    @classmethod
    def uniform(
            cls: Type['Stochastic[T]'],
            values: Iterable[T]
    ) -> 'Stochastic[T]':
        values = list(values)
        p = 1 / len(values)
        return cls((v, p) for v in values)

    def __missing__(self, key):
        return 0

    def __hash__(self) -> int:
        try:
            return self._hash
        except AttributeError:
            h = self._hash = hash(tuple(sorted(self.items())))
            return h

    def _bag(self: 'Stochastic[T]', k: int) -> StochasticSeq[Sequence[T]]:
        kf = factorial(k)
        for point in combinations_with_replacement(self.items(), k):
            v = tuple(v for v, _ in point)
            p: float = reduce(operator.mul, [p for _, p in point], 1)
            yield v, p * (kf / combinations(point))

    def bag(self: 'Stochastic[T]', k: int) -> 'Stochastic[Sequence[T]]':
        return Stochastic(self._bag(k))

    def expected(self: 'Stochastic[int]') -> float:
        fun: Reduce[float] = operator.add
        return reduce(fun, (v * p for v, p in self.items()))

    def variance(self: 'Stochastic[int]') -> float:
        mu = self.expected()
        return sum((p * (v - mu) ** 2 for v, p in self.items()))

    def stddev(self: 'Stochastic[int]') -> float:
        return sqrt(self.variance())

    def __add__(self, other: 'Stochastic[T]') -> 'Stochastic[T]':
        fun: Reduce[T] = operator.add
        return apply((self, other), fun)

    def __mul__(self, scalar: int) -> 'Stochastic[T]':
        fun: Reduce['Stochastic[T]'] = operator.add
        return reduce(fun, [self] * scalar)


# pylint: disable=unused-argument, pointless-statement, function-redefined
@overload
def apply(
    vars: Tuple[Stochastic[A0]],
    op: Callable[[A0], Union[Stochastic[T], T]]
) -> Stochastic[T]: ...


@overload
def apply(
    vars: Tuple[Stochastic[A0], Stochastic[A1]],
    op: Callable[[A0, A1], Union[Stochastic[T], T]]
) -> Stochastic[T]: ...


@overload
def apply(
    vars: Tuple[Stochastic[A0], Stochastic[A1], Stochastic[A2]],
    op: Callable[[A0, A1, A2], Union[Stochastic[T], T]]
) -> Stochastic[T]: ...


@overload
def apply(
    vars: Iterable[Stochastic[A0]],
    op: Callable[..., Union[Stochastic[T], T]]
) -> Stochastic[T]: ...


def apply(vars, op: Callable[..., Union[Stochastic[T], T]]) -> Stochastic[T]:
    result: Stochastic[T] = Stochastic()
    for point in product(*[v.items() for v in vars]):
        v = op(*[v for v, _ in point])
        p = reduce(operator.mul, [p for _, p in point], 1)
        if isinstance(v, Stochastic):
            for iv, ip in v.items():
                result[iv] += (p * ip)
        else:
            result[v] += p
    return result
