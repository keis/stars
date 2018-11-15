import re
import operator
from functools import reduce
from typing import Iterable

from stochastic import Stochastic, Reduce, apply


def sign(i: int) -> int:
    if i > 0:
        return 1
    if i < 0:
        return -1
    return 0


def versus(a: int, b: int) -> int:
    return sign(a - b)


def keep(vars: Iterable[Stochastic[int]], count: int) -> Stochastic[int]:
    fun: Reduce[int] = operator.add

    def _keep(*v: int) -> int:
        return reduce(fun, sorted(v, reverse=True)[:count])

    return apply(vars, _keep)


def dice(spec: str) -> Stochastic[int]:
    m = re.match(r'(\d)?d(\d+)(?:k(\d))?(?:\+(\d+))?', spec)
    if m is None:
        raise ValueError(f"Invalid format of dice spec `${spec}`")
    count, size, k, offset = [int(v) if v else None for v in m.groups()]
    d = Stochastic.uniform(range(1, (size or 0) + 1))
    return (
        keep([d] * (count or 1), k or count or 1)
        + Stochastic.constant(offset or 0)
    )
