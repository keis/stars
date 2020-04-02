from __future__ import annotations
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


# math.prod in >=3.8
def prod(factors: Iterable[int]) -> int:
    return reduce(operator.mul, factors)


def combinations_with_permutation_count(
        items: Iterable[T],
        n: int
) -> Iterable[Tuple[Tuple[T, ...], int]]:
    """
    Generates all the multisets of cardinality `n` from `items` and their
    number of permutations
    """
    nf: int = factorial(n)
    for item in combinations_with_replacement(items, n):
        kf = prod([factorial(len(list(grp))) for _, grp in groupby(item)])
        yield item, (nf // kf)


class Stochastic(Dict[T, float]):
    _hash: int

    @classmethod
    def constant(cls: Type[Stochastic[T]], value: T) -> Stochastic[T]:
        return cls(((value, 1),))

    @classmethod
    def uniform(
            cls: Type[Stochastic[T]],
            values: Iterable[T]
    ) -> Stochastic[T]:
        values = list(values)
        p = 1 / len(values)
        return cls((v, p) for v in values)

    @classmethod
    def collect(
            cls: Type[Stochastic[T]],
            items: Iterable[Tuple[Union[T, Stochastic[T]], float]]
    ) -> Stochastic[T]:
        result: Stochastic[T] = Stochastic()
        for (v, p) in items:
            if isinstance(v, Stochastic):
                for iv, ip in v.items():
                    result[iv] += (p * ip)
            else:
                result[v] += p
        return result

    def __missing__(self, key):
        return 0

    def __hash__(self) -> int:
        try:
            return self._hash
        except AttributeError:
            h = self._hash = hash(tuple(sorted(self.items())))
            return h

    def _bag(self: Stochastic[T], k: int) -> StochasticSeq[Sequence[T]]:
        for point, c in combinations_with_permutation_count(self.items(), k):
            v = tuple(v for v, _ in point)
            p: float = reduce(operator.mul, [p for _, p in point], 1)
            yield v, p * c

    def bag(self: Stochastic[T], k: int) -> Stochastic[Sequence[T]]:
        return Stochastic(self._bag(k))

    def map(
            self: Stochastic[T],
            fun: Callable[[T], Union[Stochastic[S], S]]
    ) -> Stochastic[S]:
        return Stochastic.collect((fun(v), p) for v, p in self.items())

    def expected(self: Stochastic[int]) -> float:
        fun: Reduce[float] = operator.add
        return reduce(fun, (v * p for v, p in self.items()))

    def variance(self: Stochastic[int]) -> float:
        mu = self.expected()
        return sum((p * (v - mu) ** 2 for v, p in self.items()))

    def stddev(self: Stochastic[int]) -> float:
        return sqrt(self.variance())

    def __add__(self, other: Stochastic[T]) -> Stochastic[T]:
        fun: Reduce[T] = operator.add
        return apply((self, other), fun)

    def __mul__(self, scalar: int) -> Stochastic[T]:
        fun: Reduce[Stochastic[T]] = operator.add
        return reduce(fun, [self] * scalar)

    def __le__(self: Stochastic[int], scalar: int) -> Stochastic[bool]:
        return self.map(lambda a: a <= scalar)


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
    def _apply():
        for point in product(*[v.items() for v in vars]):
            v = op(*[v for v, _ in point])
            p = reduce(operator.mul, [p for _, p in point], 1)
            yield v, p
    return Stochastic.collect(_apply())
