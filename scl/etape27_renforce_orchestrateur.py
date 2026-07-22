"""ÉTAPE 27 — REINFORCE l'orchestrateur au‑delà de l'imitation (§4-5 du plan).

L'imitation d'A* donne une politique RÉACTIVE (~50 %, étape 26) : elle perd la précision
multi‑pas d'A*. On la démarre à froid par imitation, PUIS on la pousse par REINFORCE sur des
épisodes réels — récompense = sucre mangé − coût des pas. C'est l'analogue nocturne : rejouer
des situations, échantillonner des actions, renforcer ce qui mange le sucre. On mesure la
visée AVANT et APRÈS renforcement.

    python3 -m scl.etape27_renforce_orchestrateur
"""
import argparse
import random

import numpy as np

from .dynamique_objet import DynamiqueObjet
from .etape26_orchestrateur_action import _donnees, _entrainer_vq, _evaluer, _former
from .maternage import placer_pres_sucre
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde
from .orchestrateur_action import tokens_objets
from .perception_objet import ChampObjet
from .planification_objet import chercher_sucre

A = ACCELERATIONS_PERMISES


def _reinforce(po, orch, episodes, rng, max_pas=6, cout_pas=0.05, taille_lot=32):
    lot = []
    for e in range(episodes):
        m = Monde(graine=40000 + e)
        if placer_pres_sucre(m, rng, dmin=2, dmax=4) is None:
            continue
        traj, got = [], 0
        for _ in range(max_pas):
            objets = po.objets(np.asarray(m.percevoir()["vision"][-1]).copy())
            a_idx, logp, ent = orch.echantillonner(tokens_objets(po, objets, orch), m.vitesse.tolist())
            traj.append((logp, ent))
            if any(ev == "sucre" for ev in m.appliquer_action(A[a_idx])):
                got = 1; break
        if not traj:
            continue
        lot.append((traj, got - cout_pas * len(traj)))
        if len(lot) >= taille_lot:                       # REINFORCE par lot, avantages standardisés
            R = np.array([r for _, r in lot], dtype=np.float32)
            mu, sd = float(R.mean()), float(R.std()) + 1e-6
            orch.pas_renforce_lot([(tr, (r - mu) / sd) for tr, r in lot])
            lot = []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=4000)
    p.add_argument("--placements", type=int, default=1500)
    p.add_argument("--episodes", type=int, default=3000)
    args = p.parse_args()
    import torch; torch.manual_seed(0); random.seed(0)
    print(f"Device : {DEVICE} — imitation d'A* puis REINFORCE")

    po = ChampObjet(); _entrainer_vq(po, args.pas)
    dyn = DynamiqueObjet(po)
    data = _donnees(po, dyn, args.placements, np.random.default_rng(0))
    orch = _former(po, dyn, data, avec_type=True)          # démarrage à froid par imitation

    graines = list(range(20000, 20250))
    ev = lambda: _evaluer(po, lambda o, v: A[orch.choisir(tokens_objets(po, o, orch), v)], graines)
    r_avant = ev()
    _reinforce(po, orch, args.episodes, np.random.default_rng(1))
    r_apres = ev()
    r_astar = _evaluer(po, lambda o, v: (chercher_sucre(po, dyn, o, v) or [(0, 0)])[0], graines)
    r_hasard = _evaluer(po, lambda o, v: random.choice(A), graines)

    print(f"\nSucre mangé (placements frais, {len(graines)} essais) :")
    print(f"   A* (professeur)              : {r_astar}%")
    print(f"   orchestrateur imitation seule: {r_avant}%")
    print(f"   orchestrateur + REINFORCE    : {r_apres}%   ← doit monter vers A*")
    print(f"   hasard                       : {r_hasard}%")
    ok = r_apres > r_avant + 5
    print(f"\n{'OK' if ok else 'à affiner'} — le REINFORCE pousse l'orchestrateur au‑delà de "
          f"l'imitation ({r_avant}%→{r_apres}%), vers A* ({r_astar}%)"
          if ok else f"\nà affiner — REINFORCE n'améliore pas nettement ({r_avant}%→{r_apres}%)")


if __name__ == "__main__":
    main()
