#!/usr/bin/env python3
"""Expérience 1 — l'animal SCL apprend-il ? Comparaison contre le hasard.

Fait tourner, sur le MÊME monde (même graine) :
  A. une politique aléatoire (baseline)
  B. l'agent SCL complet (jour/nuit)
puis compare sucres/jour et bâtons/jour.

Usage :
    python3 experience.py --jours 10 --steps 1000 --graine 42
Le run SCL écrit son audit dans exp_scl.jsonl (visualisable avec viewer.py).
"""
import argparse
import json
import random

from scl.logger import configurer
from scl.monde import Monde, ACCELERATIONS_PERMISES
from scl.memoires import TableBesoins
from scl.boucle import main_loop


def run_aleatoire(n_jours, steps, graine):
    """Baseline : accélérations tirées au hasard, mêmes règles du monde."""
    monde = Monde(graine=graine)
    besoins = TableBesoins()
    rng = random.Random(graine)
    par_jour = []
    prec = {"sucre": 0, "baton": 0}
    for jour in range(n_jours):
        for _ in range(steps):
            monde.percevoir()
            ev = monde.appliquer_action(rng.choice(ACCELERATIONS_PERMISES))
            besoins.mettre_a_jour(ev, 1)
        par_jour.append({"sucres": monde.compteurs["sucre"] - prec["sucre"],
                         "batons": monde.compteurs["baton"] - prec["baton"]})
        prec = {"sucre": monde.compteurs["sucre"],
                "baton": monde.compteurs["baton"]}
    return par_jour


def run_scl(n_jours, steps, graine, chemin_log="exp_scl.jsonl"):
    """Agent SCL complet ; métriques par jour lues dans le log d'audit."""
    open(chemin_log, "w").close()   # nouveau run propre (le viewer s'y recale)
    logger = configurer(chemin=chemin_log)
    main_loop(n_jours=n_jours, steps_par_jour=steps, graine=graine,
              verbose=True)
    logger.fermer()
    resumes = []
    with open(chemin_log, encoding="utf-8") as f:
        for ligne in f:
            try:
                rec = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            if rec.get("action") == "resume_journee":
                resumes.append(rec)
    par_jour, prec = [], {"sucres": 0, "batons": 0}
    for r in resumes:
        par_jour.append({"sucres": r["sucres"] - prec["sucres"],
                         "batons": r["batons"] - prec["batons"],
                         "erreur": r["erreur_globale"],
                         "n_modules": r["n_modules"]})
        prec = {"sucres": r["sucres"], "batons": r["batons"]}
    return par_jour


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jours", type=int, default=10)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--graine", type=int, default=42)
    args = p.parse_args()

    print(f"=== Baseline aléatoire ({args.jours} jours × {args.steps} steps) ===")
    alea = run_aleatoire(args.jours, args.steps, args.graine)
    print(f"=== Agent SCL (même monde, graine {args.graine}) ===")
    scl = run_scl(args.jours, args.steps, args.graine)

    print("\njour |  aléatoire (sucres/bâtons) |  SCL (sucres/bâtons) | erreur SCL | modules")
    for j in range(min(len(alea), len(scl))):
        a, s = alea[j], scl[j]
        print(f"{j:4d} | {a['sucres']:10d} / {a['batons']:<6d} |"
              f" {s['sucres']:9d} / {s['batons']:<6d} |"
              f" {s.get('erreur', float('nan')):10.4f} | {s.get('n_modules', '?')}")

    n = min(len(alea), len(scl))
    moitie = max(1, n // 2)
    ratio = lambda serie, cle, deb, fin: (
        sum(x[cle] for x in serie[deb:fin]) / max(1, fin - deb))
    print("\n=== Verdict ===")
    print(f"Sucres/jour  — aléatoire : {ratio(alea, 'sucres', 0, n):.1f}"
          f" | SCL 1re moitié : {ratio(scl, 'sucres', 0, moitie):.1f}"
          f" | SCL 2e moitié : {ratio(scl, 'sucres', moitie, n):.1f}")
    print(f"Bâtons/jour  — aléatoire : {ratio(alea, 'batons', 0, n):.1f}"
          f" | SCL 1re moitié : {ratio(scl, 'batons', 0, moitie):.1f}"
          f" | SCL 2e moitié : {ratio(scl, 'batons', moitie, n):.1f}")
    print("\nCritères de réussite (étape 1 de la roadmap) :")
    print(" - SCL 2e moitié > aléatoire en sucres/jour  → l'action sert l'objectif")
    print(" - SCL 2e moitié > SCL 1re moitié            → il y a apprentissage")
    print(" - erreur SCL décroissante sur les jours     → la prédiction s'améliore")
    print("Audit du run SCL : exp_scl.jsonl (python3 viewer.py --log exp_scl.jsonl)")


if __name__ == "__main__":
    main()
