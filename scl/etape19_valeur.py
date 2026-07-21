"""ÉTAPE 19 — la VALEUR apprise (g+h) fait viser le sucre lointain (§6 conception).

Q(champ, action) apprise par TD (bootstrap `r + γ·max Q`) combine g() (récompense immédiate)
et h() (valeur du reste à faire). Le bootstrap REMONTE le crédit d'une récompense lointaine
(sucre) vers les états en amont → l'agent VISE le sucre, là où le glouton 1-pas de l'étape 18
ne faisait qu'éviter la douleur immédiate. On compare l'agent-valeur au hasard (même monde).

    python3 -m scl.etape19_valeur
"""
import argparse
import random

import numpy as np

from .etape18_boucle_action import _agent_aleatoire
from .logger import set_temps
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde
from .planification import ModeleValeurQ
from .pulsions import Pulsions


def _agent_valeur(pas, graine, epsilon, gamma):
    monde = Monde(graine=graine)
    q = ModeleValeurQ(n_actions=len(ACCELERATIONS_PERMISES), gamma=gamma)
    puls = Pulsions()
    batons, sucres = [], []
    s = np.asarray(monde.percevoir()["vision"][-1]).copy()
    for step in range(pas):
        set_temps(step=step)
        a = q.choisir(s, epsilon=epsilon)
        ev = monde.appliquer_action(ACCELERATIONS_PERMISES[a])
        s2 = np.asarray(monde.percevoir()["vision"][-1]).copy()
        r = puls.recompense(evenements=ev)
        q.observer(s, a, r, s2)
        batons.append(sum(1 for e in ev if e == "baton"))
        sucres.append(sum(1 for e in ev if e == "sucre"))
        s = s2
    return np.array(batons), np.array(sucres)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=6000)
    p.add_argument("--epsilon", type=float, default=0.15)
    p.add_argument("--gamma", type=float, default=0.95)
    args = p.parse_args()
    print(f"Device : {DEVICE} — Q(champ,action) par TD (g+h), {args.pas} pas vs hasard")

    random.seed(0)
    bq, sq = _agent_valeur(args.pas, graine=7, epsilon=args.epsilon, gamma=args.gamma)
    ba, sa = _agent_aleatoire(args.pas, graine=7)

    n = args.pas
    d, f = slice(0, n // 2), slice(n // 2, n)
    tx = lambda x: 1000 * float(np.mean(x))
    print("\nSucres mangés (taux /1000 pas) :")
    print(f"   hasard      : {tx(sa):.1f}")
    print(f"   valeur début: {tx(sq[d]):.1f}   valeur fin: {tx(sq[f]):.1f}")
    print("Bâtons percutés (taux /1000 pas) :")
    print(f"   hasard      : {tx(ba):.1f}")
    print(f"   valeur début: {tx(bq[d]):.1f}   valeur fin: {tx(bq[f]):.1f}")

    gain_sucre = tx(sq[f]) - tx(sa)
    print(f"\nSucre : {tx(sq[f]):.1f} (valeur fin) vs {tx(sa):.1f} (hasard) → {gain_sucre:+.1f}/1000")
    print(f"Douleur : {tx(bq[f]):.1f} (valeur fin) vs {tx(ba):.1f} (hasard) → {tx(bq[f]) - tx(ba):+.1f}/1000")
    # Honnêteté : un écart de quelques unités/1000 est du BRUIT, pas une visée.
    MARGE = 5.0
    vise = gain_sucre > MARGE and tx(sq[f]) > tx(sq[d]) + MARGE
    if vise:
        print("\nOK — la valeur apprise fait VISER le sucre (nettement au-dessus du hasard)")
    else:
        print("\nRÉSULTAT NÉGATIF (attendu, honnête) — le TD EN LIGNE ne craque PAS la visée du "
              f"sucre : gain {gain_sucre:+.1f}/1000 dans le bruit. La récompense (contact du sucre) "
              "est trop RARE et non façonnée pour l'apprentissage en ligne. Le MÉCANISME de valeur "
              "est pourtant bon (test unitaire : le TD propage la récompense vers la bonne action) "
              "— ce qu'il manque, c'est de CONCENTRER l'apprentissage sur les rares épisodes "
              "récompensés : c'est précisément le REJEU NOCTURNE AMONT (étape 20).")


if __name__ == "__main__":
    main()
