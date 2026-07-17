"""Harnais d'évaluation objective du POC SCL (non versionné dans le cœur théorique).

Mesure ce que le projet cherche à voir émerger : capacité à manger (naviguer +
gérer la vitesse), efficacité (steps par sucre = "détours"), évitement des
bâtons, et progression de l'apprentissage (erreur globale des modules).

Usage : python3 -m scl.eval_poc --jours 10 --steps 300 --graines 1 2 3
"""
import argparse

import torch

from .boucle import EtatSCL, boucle_temps_reel, cycle_nocturne
from .inne import construire_graphe_inne
from .logger import set_temps
from .memoires import TableBesoins
from .monde import Monde


def evaluer(graine, n_jours, steps_par_jour, silencieux=True):
    graphe, discriminateur = construire_graphe_inne()
    monde = Monde(graine=graine)
    table_besoins = TableBesoins()
    etat = EtatSCL(graphe, discriminateur, monde, table_besoins)

    erreurs_par_jour = []
    sucres_par_jour = []
    batons_par_jour = []
    sucre_precedent = 0
    baton_precedent = 0

    for jour in range(n_jours):
        set_temps(jour=jour)
        for step in range(steps_par_jour):
            set_temps(step=step)
            boucle_temps_reel(etat, t=step)
        cycle_nocturne(etat, t=steps_par_jour)
        s = monde.compteurs["sucre"] - sucre_precedent
        b = monde.compteurs["baton"] - baton_precedent
        sucre_precedent = monde.compteurs["sucre"]
        baton_precedent = monde.compteurs["baton"]
        sucres_par_jour.append(s)
        batons_par_jour.append(b)
        erreurs_par_jour.append(round(graphe.erreur_globale(), 5))

    total_steps = n_jours * steps_par_jour
    total_sucre = monde.compteurs["sucre"]
    total_baton = monde.compteurs["baton"]
    steps_par_sucre = (total_steps / total_sucre) if total_sucre else float("inf")
    mp = etat.modele_prevision
    err_prev = (sum(mp.erreur_recente) / len(mp.erreur_recente)) if mp.erreur_recente else None
    n_appris = etat.compteur_mode["appris"]
    return {
        "graine": graine,
        "sucres": total_sucre,
        "batons": total_baton,
        "steps_par_sucre": round(steps_par_sucre, 1),
        "sucres_par_jour": sucres_par_jour,
        "batons_par_jour": batons_par_jour,
        "erreur_debut": erreurs_par_jour[0] if erreurs_par_jour else None,
        "erreur_fin": erreurs_par_jour[-1] if erreurs_par_jour else None,
        "n_modules": len(graphe.modules),
        "prevision_err": round(err_prev, 5) if err_prev is not None else None,
        "prevision_fiab": round(mp.fiabilite(), 3),
        "frac_appris": round(n_appris / total_steps, 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jours", type=int, default=10)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--graines", type=int, nargs="+", default=[1, 2, 3])
    args = p.parse_args()

    torch.manual_seed(0)
    resultats = []
    for g in args.graines:
        r = evaluer(g, args.jours, args.steps)
        resultats.append(r)
        print(f"[graine {g}] sucres={r['sucres']:3d} batons={r['batons']:3d} "
              f"steps/sucre={r['steps_par_sucre']:6} "
              f"err {r['erreur_debut']}→{r['erreur_fin']} "
              f"modules={r['n_modules']}")
        print(f"           sucres/jour={r['sucres_par_jour']}")
        print(f"           modèle corps: err={r['prevision_err']} "
              f"fiab={r['prevision_fiab']} frac_navig_appris={r['frac_appris']}")

    n = len(resultats)
    moy_sucre = sum(r["sucres"] for r in resultats) / n
    moy_baton = sum(r["batons"] for r in resultats) / n
    finis = [r["steps_par_sucre"] for r in resultats if r["steps_par_sucre"] != float("inf")]
    moy_sps = (sum(finis) / len(finis)) if finis else float("inf")
    print(f"\n=== MOYENNE ({n} graines) : sucres={moy_sucre:.1f} "
          f"batons={moy_baton:.1f} steps/sucre={moy_sps:.1f} ===")


if __name__ == "__main__":
    main()
