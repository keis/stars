import re
import operator
from functools import reduce
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
        for point in product(*[v.items() for v in [self, *other]]):
            v = op(*[v for v, _ in point])
            p = reduce(operator.mul, [p for _, p in point], 1)
            result[v] += p
        return result

    def __add__(self, other: 'Stochastic'):
        return self.apply([other], operator.add)

    def __mul__(self, scalar: int):
        return reduce(operator.add, [self] * scalar)


def keep(vars, count):
    head, *others = vars
    return head.apply(
        others,
        lambda *v: reduce(operator.add, sorted(v, reverse=True)[:count])
    )


def dice(spec) -> Stochastic:
    m = re.match(r'(\d)?d(\d+)(?:k(\d))?(?:\+(\d+))?', spec)
    count, size, k, offset = [int(v) if v else None for v in m.groups()]
    d = Stochastic.uniform(range(1, size + 1))
    return (
        keep([d] * (count or 1), k or count or 1)
        + Stochastic.constant(offset or 0)
    )


def attack(attacker, defender, damage):
    return attacker.apply(
        [defender, damage], lambda a, b, c: c if a >= b else 0)


if __name__ == '__main__':
    atk = attack(dice('d10+2'), dice('d10'), dice('2d8+2'))
    print(atk)
