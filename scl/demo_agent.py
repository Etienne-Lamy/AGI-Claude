"""DÉMO AGENT (étape 28) — VOIR l'agent objet : percevoir, prédire, planifier, agir.

Produit un log JSONL pour le viewer : champ VU vs champ PRÉDIT (modèle‑objet, décalage par la
vitesse), objets suivis, vitesse, plan A* en cours, action poussée, sucres/bâtons cumulés.

    python3 -m scl.demo_agent --log agent.jsonl --pas 400
    python3 viewer.py --log agent.jsonl --port 8400     # 2e terminal → panneau « agent »
"""
import argparse
import random

import numpy as np

from .dynamique_objet import DynamiqueObjet
from .logger import configurer, log, set_temps
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde
from .perception_objet import ChampObjet
from .planification_objet import chercher_sucre

VITESSES = [(1, 0), (2, 0), (0, 1), (1, 1), (0, 0), (-1, 0), (0, -1)]


def _grille(x, h, w):
    a = np.asarray(x, dtype=float).reshape(h, w)
    return [[round(float(v), 3) for v in ligne] for ligne in a]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="agent.jsonl")
    p.add_argument("--pas", type=int, default=400)
    p.add_argument("--pas_vq", type=int, default=3000)
    args = p.parse_args()
    random.seed(0)
    import torch; torch.manual_seed(0)
    configurer(chemin=args.log, verbeux=False)
    print(f"Log : {args.log} (device {DEVICE}) — lance `python3 viewer.py --log {args.log}` en parallèle")

    po = ChampObjet()
    log("viewer", "phase", nom="apprentissage_vision", info="le VQ apprend les catégories")
    m = Monde(graine=1); rng = np.random.default_rng(0)
    for s in range(args.pas_vq):                       # apprentissage perception (VQ)
        set_temps(step=s)
        po.entrainer(np.asarray(m.percevoir()["vision"][-1]).copy())
        if s % 15 == 0:
            m.vitesse = np.array(VITESSES[int(rng.integers(len(VITESSES)))], dtype=np.int64)
        m.appliquer_action((0, 0))
    po.calibrer()
    dyn = DynamiqueObjet(po)

    log("viewer", "meta", h=po.t, w=po.t)
    log("viewer", "phase", nom="agent", info="perçoit → prédit → planifie A* → agit")
    monde = Monde(graine=2025); monde.vitesse = np.zeros(2, np.int64)
    sucres = batons = 0
    for step in range(args.pas):
        set_temps(step=step)
        champ = np.asarray(monde.percevoir()["vision"][-1]).copy()
        objets = po.objets(champ)
        prevu = po.regenerer(po.decaler(objets, monde.vitesse.tolist()))   # prédiction-objet T+1
        plan = chercher_sucre(po, dyn, objets, v=monde.vitesse.tolist())
        if plan:
            a = plan[0]; mode = "plan"                 # viser le sucre
        else:
            a = random.choice(ACCELERATIONS_PERMISES); mode = "explore"    # rien en vue → explorer
        log("viewer", "champ", vu=_grille(champ, po.t, po.t), prevu=_grille(prevu, po.t, po.t),
            rappel=1.0, regime=f"v={monde.vitesse.tolist()}", module="modèle-objet")
        log("viewer", "agent", vitesse=monde.vitesse.tolist(), acte=list(a), mode=mode,
            plan=[list(x) for x in (plan or [])], n_objets=len(objets),
            objets=[[int(k), int(i), int(j)] for k, (i, j) in objets],
            sucres=sucres, batons=batons)
        ev = monde.appliquer_action(a)
        sucres += sum(1 for e in ev if e == "sucre")
        batons += sum(1 for e in ev if e == "baton")
    print(f"Démo terminée — {sucres} sucres, {batons} bâtons en {args.pas} pas. Log complet.")


if __name__ == "__main__":
    main()
