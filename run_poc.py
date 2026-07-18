#!/usr/bin/env python3
"""Point d'entrée du POC SCL.

Usage (Ubuntu/WSL) :
    python3 run_poc.py --jours 3 --steps 500 --log scl_audit.jsonl
    python3 run_poc.py --jours 1 --steps 100 --console   # audit à l'écran

Le log d'audit (JSONL, une action par ligne) permet de suivre chaque action
de chaque module : création, gate, entraînement, condensateurs, verrous,
ruptures, découpes, rêves, clustering...
"""
import argparse

from scl.logger import configurer
from scl.boucle import main_loop
from scl import curiosite


def main():
    p = argparse.ArgumentParser(description="POC SCL — petit animal virtuel")
    p.add_argument("--jours", type=int, default=3)
    p.add_argument("--steps", type=int, default=500, help="steps par jour")
    p.add_argument("--log", type=str, default="scl_audit.jsonl",
                   help="fichier d'audit JSONL ('' pour désactiver)")
    p.add_argument("--console", action="store_true",
                   help="affiche aussi l'audit sur stdout")
    p.add_argument("--verbeux", action="store_true",
                   help="journalise aussi les forwards/gates/buffers "
                        "(log ~50x plus gros, pour audit fin)")
    p.add_argument("--graine", type=int, default=None)
    p.add_argument("--delai", type=float, default=0.0,
                   help="secondes de pause par step (ex: 0.3 pour observer "
                        "chaque mouvement dans le dashboard)")
    p.add_argument("--checkpoint", type=str, default="cerveau.pkl",
                   help="fichier d'état persistant (défaut: cerveau.pkl) — "
                        "repris s'il existe, sauvegardé chaque nuit. "
                        "Passer --checkpoint '' pour repartir de zéro.")
    args = p.parse_args()

    logger = configurer(chemin=args.log or None, console=args.console,
                        verbeux=args.verbeux)
    if args.delai > 0:
        from scl.config import CONFIG
        CONFIG["delai_step"] = args.delai
    etat = main_loop(n_jours=args.jours, steps_par_jour=args.steps,
                     graine=args.graine, verbose=True,
                     checkpoint=args.checkpoint or None)
    monde = etat.monde
    dyn = etat.dynamique
    print("\n=== Bilan final ===")
    print(f"Modules : {sorted(etat.graphe.modules)}")
    if "vision" in etat.graphe.modules:
        print(f"Incertitude vision (champ statique) : "
              f"{curiosite.incertitude(etat.graphe.modules['vision']):.4f}")
    print(f"Prédicteurs de dynamique émergés : {len(dyn.predicteurs)} "
          f"(accélérations : {sorted(list(a) for a in dyn.predicteurs)})")
    rapport = dyn.etat_maitrise()
    n_maitrises = sum(1 for _, (_, m) in rapport.items() if m)
    print(f"Accélérations maîtrisées : {n_maitrises}/{len(rapport)}  détail={rapport}")
    print(f"(Contexte secondaire — sucres incidents : {monde.compteurs['sucre']}, "
          f"bâtons : {monde.compteurs['baton']})")
    if args.checkpoint:
        print(f"Cerveau sauvegardé : {args.checkpoint} (relancer pour continuer)")
    if args.log:
        print(f"Audit : {args.log} ({logger.n} actions journalisées)")
    logger.fermer()


if __name__ == "__main__":
    main()
