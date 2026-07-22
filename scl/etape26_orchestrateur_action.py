"""ÉTAPE 26 — l'orchestrateur à ATTENTION apprend d'A* et AMORTIT la recherche (§4 du plan).

A* (`chercher_sucre`) est le PROFESSEUR : il fournit la première action optimale de chaque
état le long de ses plans. L'orchestrateur (Set Transformer sur les jetons‑objets + signaux
faibles) l'IMITE (entropie croisée). À l'évaluation il émet l'action SANS dérouler l'arbre
→ il amortit A*. Ablation : retirer le SIGNAL de type d'objet (sucre vs bâton) doit dégrader
(preuve que les signaux faibles servent).

    python3 -m scl.etape26_orchestrateur_action
"""
import argparse
import random

import numpy as np

from .dynamique_objet import DynamiqueObjet
from .logger import set_temps
from .maternage import placer_pres_sucre
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde
from .orchestrateur_action import OrchestrateurAction, tokens_objets
from .perception_objet import ChampObjet
from .planification_objet import chercher_sucre

VITESSES_ENTRAINEMENT = [(1, 0), (2, 0), (0, 1), (1, 1), (0, 0), (-1, 0), (0, -1)]
IDX = {a: i for i, a in enumerate(ACCELERATIONS_PERMISES)}


def _entrainer_vq(po, pas):
    m = Monde(graine=1); rng = np.random.default_rng(0)
    for s in range(pas):
        set_temps(step=s)
        po.entrainer(np.asarray(m.percevoir()["vision"][-1]).copy())
        if s % 15 == 0:
            m.vitesse = np.array(VITESSES_ENTRAINEMENT[int(rng.integers(len(VITESSES_ENTRAINEMENT)))], dtype=np.int64)
        m.appliquer_action((0, 0))
    po.calibrer()


def _donnees(po, dyn, n, rng):
    """Paires (objets, v, action*) le long des plans d'A* — l'expert montre le chemin."""
    data = []
    for e in range(n):
        m = Monde(graine=5000 + e)
        if placer_pres_sucre(m, rng, dmin=2, dmax=4) is None:
            continue
        objets = po.objets(np.asarray(m.percevoir()["vision"][-1]).copy())
        plan = chercher_sucre(po, dyn, objets, v=(0, 0))
        if not plan:
            continue
        E, v = objets, (0, 0)
        for a in plan:
            data.append((E, v, IDX[a]))
            E, v = dyn.transition(E, v, a)
    return data


def _evaluer(po, choisir, graines, pas=6):
    mange = 0
    for g in graines:
        m = Monde(graine=g); rng = np.random.default_rng(g)
        if placer_pres_sucre(m, rng, dmin=2, dmax=4) is None:
            continue
        for _ in range(pas):
            objets = po.objets(np.asarray(m.percevoir()["vision"][-1]).copy())
            a = choisir(objets, m.vitesse.tolist())
            if any(ev == "sucre" for ev in m.appliquer_action(a)):
                mange += 1; break
    return 100 * mange // len(graines)


def _former(po, dyn, data, avec_type):
    orch = OrchestrateurAction(po.clf.K, avec_type=avec_type)
    for ep in range(8):
        random.shuffle(data)
        for i in range(0, len(data), 64):
            batch = data[i:i + 64]
            lot = [(tokens_objets(po, o, orch), v, a) for o, v, a in batch]
            orch.imiter(lot)
    return orch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=4000)
    p.add_argument("--placements", type=int, default=1500)
    args = p.parse_args()
    import torch; torch.manual_seed(0); random.seed(0)
    print(f"Device : {DEVICE} — orchestrateur à attention appris d'A*")

    po = ChampObjet(); _entrainer_vq(po, args.pas)
    dyn = DynamiqueObjet(po)
    data = _donnees(po, dyn, args.placements, np.random.default_rng(0))
    print(f"   {len(data)} paires (état, action*) extraites des plans d'A*")

    orch = _former(po, dyn, data, avec_type=True)
    orch_abl = _former(po, dyn, list(data), avec_type=False)     # ablation : sans le type d'objet

    graines = list(range(20000, 20250))
    A = ACCELERATIONS_PERMISES
    r_orch = _evaluer(po, lambda o, v: A[orch.choisir(tokens_objets(po, o, orch), v)], graines)
    r_abl = _evaluer(po, lambda o, v: A[orch_abl.choisir(tokens_objets(po, o, orch_abl), v)], graines)
    r_astar = _evaluer(po, lambda o, v: (chercher_sucre(po, dyn, o, v) or [(0, 0)])[0], graines)
    r_hasard = _evaluer(po, lambda o, v: random.choice(ACCELERATIONS_PERMISES), graines)

    print(f"\nSucre mangé (placements frais, {len(graines)} essais) :")
    print(f"   A* (recherche, professeur)      : {r_astar}%")
    print(f"   orchestrateur appris (SANS recherche): {r_orch}%   ← amortit A*")
    print(f"   orchestrateur SANS signal de type   : {r_abl}%   ← ablation")
    print(f"   hasard                          : {r_hasard}%")
    ok = r_orch > r_hasard + 20 and r_orch > r_abl + 10
    print(f"\n{'OK' if ok else 'à affiner'} — l'orchestrateur à attention IMITE A* et vise le "
          f"sucre sans recherche ; retirer le signal de type DÉGRADE (les signaux faibles servent)"
          if ok else "\nà affiner")


if __name__ == "__main__":
    main()
