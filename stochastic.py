import re
import operator
import csv
from functools import reduce
from math import sqrt
from typing import Any, Iterable
from itertools import product, chain

import matplotlib.pyplot as plt
from tabulate import tabulate


def sign(i: int) -> int:
    if i > 0:
        return 1
    if i < 0:
        return -1
    return 0


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

    def stddev(self):
        mu = self.expected()
        return sqrt(
            reduce(
                operator.add,
                (p * (v - mu) ** 2 for v, p in self.items())
            ))

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


def parse_attack(attack):
    roll, dmg = attack.split(', ')
    return (*re.split('[vV]', roll), dice(dmg))


def asset_attack(asset, defender):
    if asset['Attack'] == 'None' or asset['Attack'].endswith('Special'):
        return {
            'hit': Stochastic.constant(0),
            'damage': Stochastic.constant(0),
        }

    attacker = factions[asset['Owner']]
    a, d, dmg = parse_attack(asset['Attack'])

    a_dice = dice('d10+' + attacker[a])
    d_dice = dice('d10+' + defender[d])
    return {
        'hit': a_dice.apply([d_dice], lambda a, b: sign(a - b)),
        'damage': dmg,
    }


def display(attack):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.canvas.set_window_title(
        f"{attack['attacker']} attacking {attack['defender']}")
    items = sorted(attack['damage'].items())
    keys = ('hit', 'counter', 'damage')
    ax1.bar(
        keys,
        [attack[k].expected() for k in keys],
        yerr=[attack[k].stddev() for k in keys],
    )
    ax2.bar(
        [v for v, _ in items],
        [p*100 for _, p in items]
    )


def faction_ball(name):
    def isattacking(asset):
        return not (
            asset['Attack'] == 'None' or
            asset['Attack'].endswith('Special')
        )
    fa = [a for a in assets if a['Owner'] == name]
    if any(a['Asset'] == 'Transit Web' for a in fa):
        return [
            a for a in fa
            if a['W/C/F'] in ('Wealth', 'Cunning')
            and isattacking(a)
            and a['Asset'] != 'Transit Web'
        ]
    if any(a['Asset'] == 'Covert Transit Net' for a in fa):
        return [
            a for a in fa
            if a['Type'] in ('Special Forces',) and isattacking(a)
        ]
    return []


def potential(ball, defender):
    attacks = [asset_attack(asset, defender) for asset in ball]
    return {
        'assets': [asset['Asset'] for asset in ball],
        'attacker': ball[0]['Owner'],
        'defender': defender['Faction Name'],
        'damage': reduce(
            operator.add,
            [
                a['hit'].apply([a['damage']], lambda a, d: d if a >= 0 else 0)
                for a in attacks
            ]),
        'hit': reduce(
            operator.add,
            [
                a['hit'].apply([], lambda a: int(a >= 0))
                for a in attacks
            ]),
        'counter': reduce(
            operator.add,
            [
                a['hit'].apply([], lambda a: int(a <= 0))
                for a in attacks
            ]),
    }


with open('factions.csv') as factionscsv:
    factions = {r['Faction Name']: r for r in csv.DictReader(factionscsv)}

with open('assets.csv') as assetscsv:
    assets = list(csv.DictReader(assetscsv))


def top():
    balls = filter(None, (faction_ball(f) for f in factions))
    for ball in balls:
        print(ball[0]['Owner'], '\t', ', '.join(a['Asset'] for a in ball))
    attacks = [
        potential(ball, faction)
        for faction, ball in product(factions.values(), balls)
    ]
    biggest = sorted(
        attacks, reverse=True, key=lambda a: a['damage'].expected())
    print(tabulate([
        [
            attack['attacker'],
            attack['defender'],
            str(attack['damage'].expected()),
            ', '.join(attack['assets'])
        ]
        for attack in biggest[:50]
    ]))


def details(args):
    *attackers, defender = args
    defender = factions[defender]
    ball = list(chain(*[faction_ball(attacker) for attacker in attackers]))
    attack = potential(ball, defender)
    hp = int(defender['HP']) + 15 + 10
    o = sum(p for v, p in attack['damage'].items() if v >= hp)
    print('one turn kill', o * 100)
    print(
        str(attack['damage'].expected()),
        ', '.join(attack['assets'])
    )
    print(attack['counter'].expected())
    display(attack)
    plt.show()


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'details':
        details(sys.argv[2:])
    elif sys.argv[1] == 'top':
        top()
