import operator
from functools import reduce
from math import sqrt
from typing import Any, Iterable
from itertools import product


class Stochastic(dict):
    @classmethod
    def constant(cls, value: Any) -> 'Stochastic':
        return cls(((value, 1),))

    @classmethod
    def uniform(cls, values: Iterable[Any]) -> 'Stochastic':
        values = list(values)
        p = 1 / len(values)
        return cls((v, p) for v in values)

    def __missing__(self, key):
        return 0

    def expected(self):
        return reduce(operator.add, (v * p for v, p in self.items()))

    def variance(self):
        mu = self.expected()
        return sum((p * (v - mu) ** 2 for v, p in self.items()))

    def stddev(self):
        return sqrt(self.variance())

    def apply(self, other, op):
        result = Stochastic()
        for point in product(*[v.items() for v in [self, *other]]):
            v = op(*[v for v, _ in point])
            p = reduce(operator.mul, [p for _, p in point], 1)
            if isinstance(v, Stochastic):
                for iv, ip in v.items():
                    result[iv] += (p * ip)
            else:
                result[v] += p
        return result

    def __add__(self, other: 'Stochastic'):
        return self.apply([other], operator.add)

    def __mul__(self, scalar: int):
        return reduce(operator.add, [self] * scalar)
