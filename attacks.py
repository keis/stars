import re
import operator
import csv
from argparse import ArgumentParser
from functools import reduce
from itertools import product, chain

import matplotlib.pyplot as plt
from tabulate import tabulate
from dataclasses import dataclass

from stochastic import Stochastic, apply


def sign(i: int) -> int:
    if i > 0:
        return 1
    if i < 0:
        return -1
    return 0


@dataclass(unsafe_hash=True)
class Attacker:
    asset: str
    hp: int
    hit: Stochastic
    damage: Stochastic


@dataclass(unsafe_hash=True)
class Defender:
    asset: str
    hp: int
    damage: Stochastic


def keep(vars, count):
    head, *others = vars
    return apply(
        vars,
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
    return apply(
        [attacker, defender, damage], lambda a, b, c: c if a >= b else 0)


def parse_attack(attack):
    roll, dmg = attack.split(', ')
    return (*re.split('[vV]', roll), dice(dmg))


def asset_attack(asset, defender):
    if asset['Attack'] == 'None' or asset['Attack'].endswith('Special'):
        return Attacker(
            asset=asset['Asset'],
            hp=asset['HP'],
            hit=Stochastic.constant(0),
            damage=Stochastic.constant(0),
        )

    attacker = factions[asset['Owner']]
    a, d, dmg = parse_attack(asset['Attack'])

    a_dice = dice('d10+' + attacker[a])
    d_dice = dice('d10+' + defender[d])
    return Attacker(
        asset=asset['Asset'],
        hp=int(asset['HP']),
        hit=apply([a_dice, d_dice], lambda a, b: sign(a - b)),
        damage=dmg,
    )


def asset_counter(asset):
    if asset['Counter'] == 'None':
        dmg = Stochastic.constant(0)
    else:
        dmg = dice(asset['Counter'])
    return Defender(
        asset=asset['Asset'],
        hp=int(asset['HP']),
        damage=dmg,
    )


def display(attack):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.canvas.set_window_title(
        f"{attack['attacker']} attacking {attack['defender']}")

    keys = (
        'hit', 'counter',
        'remaining_defending_assets', 'remaining_attacking_assets'
    )
    ax1.bar(
        keys,
        [attack[k].expected() for k in keys],
        yerr=[attack[k].stddev() for k in keys],
    )

    keys = ('damage', 'hplost', 'counter_damage')
    ax2.bar(
        keys,
        [attack[k].expected() for k in keys],
        yerr=[attack[k].stddev() for k in keys],
    )

    damage = sorted(attack['hplost'].items())
    ax3.bar(
        [v for v, _ in damage],
        [p*100 for _, p in damage]
    )

    counterdamage = sorted(attack['counter_damage'].items())
    ax4.bar(
        [v for v, _ in counterdamage],
        [p*100 for _, p in counterdamage]
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
    if any(a['Asset'] == 'Covert Shipping' for a in fa):
        return [
            a for a in fa
            if a['Type'] in ('Special Forces',) and isattacking(a)
        ][:1]
    return []


def on_world(faction, world):
    return [
        a for a in assets
        if a['Owner'] == faction['Faction Name']
        and a['Location'] == world
    ]


def main_boi(faction):
    return max([
        a for a in assets
        if a['Owner'] == faction['Faction Name']
        and a['Asset'] == 'Base Of Influence'
    ], key=lambda boi: int(boi['HP']))


def main_boi_defense(faction):
    return on_world(faction, main_boi(faction)['Location'])


def order_attacks(attacks):
    return sorted(
        attacks,
        key=lambda atk: (atk.hit.expected(), atk.damage.expected()),
        reverse=True,
    )


def order_defense(counters):
    return sorted(
        counters,
        key=lambda c: (
            c.asset == 'Base Of Influence',
            c.damage[0] == 1,
            c.damage.expected()
        )
    )


def apply_damage(assets, dmg):
    if not assets:
        return assets
    head, *tail = assets
    if head > dmg:
        return (head - dmg, *tail)
    return tuple(tail)


def apply_damage_v2(assets, dmg):
    """The cheesy damage apply

    In some situations we want to let the damage go to the BoI rather than the
    defending asset. If ANY of the following is true the asset takes the hit

    * The asset would survive the damage
    * The BoI would NOT survive the damage
    * The damage is 10 or more

    Otherwise, the damage goes to the BoI
    """

    if not assets:
        return assets
    head = assets[0]
    boi = assets[-1]
    if head > dmg:
        return (head - dmg, *assets[1:])
    if boi > dmg and dmg <= 100:
        return (*assets[:-1], boi - dmg)
    return tuple(assets[1:])


def potential(attackers, defenders):
    defender = factions[defenders[0]['Owner']]
    attacks = order_attacks(
        asset_attack(asset, defender) for asset in attackers)
    defense = order_defense(
        asset_counter(asset) for asset in defenders)

    damage = []
    counter_damage = []
    remaining_defenders = Stochastic.constant(tuple(a.hp for a in defense))
    remaining_attackers = Stochastic.constant(len(attacks))
    for atk in attacks:
        dmg = apply([atk.hit, atk.damage], lambda a, d: d if a >= 0 else 0)
        counter = apply(
            [remaining_defenders, atk.hit],
            lambda defenders, hit: (
                defense[-len(defenders)].damage
                if hit <= 0 and defenders else 0
            )
        )
        remaining_defenders = apply(
            [remaining_defenders, dmg], apply_damage_v2)
        remaining_attackers = apply(
            [remaining_attackers, counter],
            lambda attackers, dmg: attackers - (dmg > atk.hp))
        damage.append(dmg)
        counter_damage.append(counter)

    return {
        'attacking_assets': [attack.asset for attack in attacks],
        'defending_assets': [counter.asset for counter in defense],
        'remaining_defending_assets': apply(
            [remaining_defenders], lambda d: len(d)),
        'remaining_attacking_assets': remaining_attackers,
        'attacker': attackers[0]['Owner'],
        'defender': defender['Faction Name'],
        'damage': reduce(operator.add, damage),
        'counter_damage': reduce(operator.add, counter_damage),
        'hplost': apply(
            [remaining_defenders],
            lambda d:  sum(a.hp for a in defense) - sum(d)),
        'hit': reduce(
            operator.add,
            [
                apply([a.hit], lambda a: int(a >= 0))
                for a in attacks
            ]),
        'counter': reduce(
            operator.add,
            [
                apply([a.hit], lambda a: int(a <= 0))
                for a in attacks
            ]),
    }


with open('factions.csv') as factionscsv:
    factions = {r['Faction Name']: r for r in csv.DictReader(factionscsv)}

with open('assets.csv') as assetscsv:
    assets = list(csv.DictReader(assetscsv))


def top(args):
    balls = list(filter(None, (faction_ball(f) for f in factions)))
    for ball in balls:
        print(ball[0]['Owner'], '\t', ', '.join(a['Asset'] for a in ball))
    attacks = [
        potential(ball, main_boi_defense(faction))
        for faction, ball in product(factions.values(), balls)
    ]
    biggest = sorted(
        attacks, reverse=True, key=lambda a: a['damage'].expected())
    print(tabulate([
        [
            attack['attacker'],
            attack['defender'],
            str(attack['damage'].expected()),
            ', '.join(attack['attacking_assets'])
        ]
        for attack in biggest[:50]
    ]))


def details(args):
    attackers, defender = args.attacker, args.defender
    defender = factions[defender]
    ball = list(chain(*[faction_ball(attacker) for attacker in attackers]))
    defenders = main_boi_defense(defender)
    attack = potential(ball, defenders)
    print('one turn kill', attack['remaining_defending_assets'][0])
    print(
        'damage',
        str(attack['damage'].expected()),
        'hplost',
        str(attack['hplost'].expected())
    )
    print(
        ', '.join(attack['attacking_assets']),
        'VS',
        ', '.join(attack['defending_assets']),
    )
    print(
        "remaining attackers", attack['remaining_attacking_assets'].expected(),
        len(ball)
    )
    print(
        "remaining defenders", attack['remaining_defending_assets'].expected(),
        len(defenders)
    )
    display(attack)
    plt.show()


def main():
    parser = ArgumentParser()
    commands = parser.add_subparsers()

    topparser = commands.add_parser('top')
    topparser.set_defaults(func=top)

    detailsparser = commands.add_parser('details')
    detailsparser.add_argument('attacker', nargs='*')
    detailsparser.add_argument('defender')
    detailsparser.set_defaults(func=details)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
