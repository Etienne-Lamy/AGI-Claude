"""ÉTAPE 25 — A* sur l'état‑objet : prioriser la branche qui mange le sucre (§3 du plan).

On place l'agent près d'un sucre (2‑4 cases), A* cherche sur l'état‑objet la séquence
d'accélérations qui l'amène sur le corps, puis on EXÉCUTE ce plan dans le VRAI monde pour
vérifier qu'il mange bien le sucre (le modèle‑objet colle‑t‑il à la réalité ?). Comparaison :
agent aléatoire, même budget de pas.

    python3 -m scl.etape25_planif_objet
"""
import argparse
import random

import numpy as np

from .dynamique_objet import DynamiqueObjet
from .logger import set_temps
from .maternage import placer_pres_sucre
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde
from .perception_objet import ChampObjet
from .planification_objet import chercher_sucre

VITESSES_ENTRAINEMENT = [(1, 0), (2, 0), (0, 1), (1, 1), (0, 0), (-1, 0), (0, -1)]


def _entrainer(po, pas):
    m = Monde(graine=1); rng = np.random.default_rng(0)
    for s in range(pas):
        set_temps(step=s)
        po.entrainer(np.asarray(m.percevoir()["vision"][-1]).copy())
        if s % 15 == 0:
            m.vitesse = np.array(VITESSES_ENTRAINEMENT[int(rng.integers(len(VITESSES_ENTRAINEMENT)))], dtype=np.int64)
        m.appliquer_action((0, 0))
    po.calibrer()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=4000)
    p.add_argument("--essais", type=int, default=200)
    args = p.parse_args()
    import torch; torch.manual_seed(0); random.seed(0)
    print(f"Device : {DEVICE} — planification A* sur l'état‑objet")

    po = ChampObjet(); _entrainer(po, args.pas)
    dyn = DynamiqueObjet(po)
    rng = np.random.default_rng(0)

    trouve = mange_plan = mange_hasard = n = 0
    longueurs = []
    for e in range(args.essais):
        m = Monde(graine=1000 + e)
        if placer_pres_sucre(m, rng, dmin=2, dmax=4) is None:
            continue
        n += 1
        objets = po.objets(np.asarray(m.percevoir()["vision"][-1]).copy())
        plan = chercher_sucre(po, dyn, objets, v=(0, 0))
        if plan:
            trouve += 1; longueurs.append(len(plan))
            # exécuter le plan dans le VRAI monde
            m2 = Monde(graine=1000 + e); m2.agent_pos = m.agent_pos.copy(); m2.vitesse = np.zeros(2, np.int64)
            mange = False
            for a in plan:
                if any(ev == "sucre" for ev in m2.appliquer_action(a)):
                    mange = True; break
            mange_plan += int(mange)
        # baseline : agent aléatoire, même monde, ~6 pas
        mh = Monde(graine=1000 + e); mh.agent_pos = m.agent_pos.copy(); mh.vitesse = np.zeros(2, np.int64)
        for _ in range(6):
            if any(ev == "sucre" for ev in mh.appliquer_action(random.choice(ACCELERATIONS_PERMISES))):
                mange_hasard += 1; break

    print(f"\nSur {n} placements près d'un sucre :")
    print(f"   A* trouve un plan        : {100 * trouve // n}%  (longueur moyenne {np.mean(longueurs):.1f} actions)")
    print(f"   plan exécuté → sucre mangé: {100 * mange_plan // n}%  (le modèle‑objet colle à la réalité)")
    print(f"   agent ALÉATOIRE (6 pas)   : {100 * mange_hasard // n}%")
    ok = mange_plan > mange_hasard and trouve > 0.8 * n
    print(f"\n{'OK' if ok else 'à affiner'} — A* sur l'état‑objet PRIORISE la branche‑sucre et "
          f"son plan mange le sucre dans le vrai monde, bien au‑dessus du hasard"
          if ok else "\nà affiner")


if __name__ == "__main__":
    main()
