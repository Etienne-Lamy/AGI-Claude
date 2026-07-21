"""ÉTAPE 18 — déroulé continu : le g() appris pilote l'action (§5-6 conception).

Boucle temps réel (type MPC, ici horizon 1) : à chaque pas l'agent PERÇOIT, choisit
l'action de meilleure récompense PRÉDITE (`ModeleRecompense`, le g() de A*), l'exécute,
observe la vraie récompense et entraîne son modèle en ligne. On compare à un agent
ALÉATOIRE dans le même monde. Attendu (motivé par le constat de l'étape 17 : 76 % en
douleur au hasard) : l'agent planifié apprend à ÉVITER la douleur → moins de bâtons que le
hasard, et de plus en plus au fil du run. Viser le sucre LOINTAIN demande le multi-pas
(A*, étape 19) + le crédit nocturne amont (étape 20).

    python3 -m scl.etape18_boucle_action
"""
import argparse
import random

import numpy as np

from .logger import set_temps
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde
from .planification import ModeleRecompense, choisir_glouton
from .pulsions import Pulsions


def _agent_planifie(pas, graine, epsilon):
    monde = Monde(graine=graine)
    modele = ModeleRecompense(n_actions=len(ACCELERATIONS_PERMISES))
    puls = Pulsions()
    batons, sucres = [], []      # 1/0 par pas (pour découper début/fin)
    for s in range(pas):
        set_temps(step=s)
        champ = np.asarray(monde.percevoir()["vision"][-1]).copy()
        a_idx = choisir_glouton(modele, champ, len(ACCELERATIONS_PERMISES), epsilon=epsilon)
        ev = monde.appliquer_action(ACCELERATIONS_PERMISES[a_idx])
        r = puls.recompense(evenements=ev)
        modele.observer(champ, a_idx, r)
        batons.append(sum(1 for e in ev if e == "baton"))
        sucres.append(sum(1 for e in ev if e == "sucre"))
    return np.array(batons), np.array(sucres)


def _agent_aleatoire(pas, graine):
    monde = Monde(graine=graine)
    batons, sucres = [], []
    for _ in range(pas):
        ev = monde.appliquer_action(random.choice(ACCELERATIONS_PERMISES))
        batons.append(sum(1 for e in ev if e == "baton"))
        sucres.append(sum(1 for e in ev if e == "sucre"))
    return np.array(batons), np.array(sucres)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=4000)
    p.add_argument("--epsilon", type=float, default=0.1)
    args = p.parse_args()
    print(f"Device : {DEVICE} — boucle MPC horizon 1, {args.pas} pas (g() appris vs hasard)")

    random.seed(0)
    bp, sp = _agent_planifie(args.pas, graine=7, epsilon=args.epsilon)
    ba, sa = _agent_aleatoire(args.pas, graine=7)

    n = args.pas
    d, f = slice(0, n // 2), slice(n // 2, n)
    tx = lambda x: 1000 * float(np.mean(x))         # taux pour 1000 pas
    print("\nBâtons percutés (taux /1000 pas) :")
    print(f"   hasard        : {tx(ba):.1f}")
    print(f"   planifié début: {tx(bp[d]):.1f}   planifié fin: {tx(bp[f]):.1f}")
    print("Sucres mangés (taux /1000 pas) :")
    print(f"   hasard        : {tx(sa):.1f}")
    print(f"   planifié début: {tx(sp[d]):.1f}   planifié fin: {tx(sp[f]):.1f}")

    reduction = tx(ba) - tx(bp[f])
    print(f"\nRéduction de douleur (fin planifié vs hasard) : {reduction:+.1f} bâtons/1000 "
          f"({100 * reduction / max(1e-9, tx(ba)):.0f}%)")
    ok = tx(bp[f]) < tx(ba) and tx(bp[f]) <= tx(bp[d])
    print("\nOK — le g() appris fait ÉVITER la douleur (moins de bâtons que le hasard, "
          "et décroissant) — première action réfléchie" if ok else
          "\nà affiner — l'évitement de la douleur n'émerge pas")


if __name__ == "__main__":
    main()
