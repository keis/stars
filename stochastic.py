import operator
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

    def apply(self, other, op):
        result = Stochastic()
        for (av, ap), (bv, bp) in product(self.items(), other.items()):
            v = op(av, bv)
            p = ap * bp
            result[v] += p
        return result

    def __add__(self, other: 'Stochastic'):
        return self.apply(other, operator.add)

    def __mul__(self, scalar: int):
        v = self
        for _ in range(1, scalar):
            v += self
        return v


if __name__ == '__main__':
    a = Stochastic.uniform(range(1, 7))
    b = Stochastic.uniform(range(1, 7))
    print((a + b).items())
    print((a * 2).items())
    print((a + Stochastic.constant(3)).items())
