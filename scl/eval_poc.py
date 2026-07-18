"""Harnais d'évaluation ORIENTÉ ÉMERGENCE du POC SCL.

Ne mesure plus « combien de sucres » (conséquence secondaire) mais la thèse :
1. l'incertitude de la vision (reconstruction du champ statique) DESCEND ;
2. une fois la vision maîtrisée, l'agent se met à AGIR (part de pas immobiles ↓) ;
3. des prédicteurs de dynamique ÉMERGENT (créés sur surprise, un par accélération) ;
4. leur incertitude descend à son tour → l'agent maîtrise son corps, région par
   région (nb d'accélérations maîtrisées ↑).

Usage : python3 -m scl.eval_poc --jours 15 --steps 300 --graines 1 2 3
"""
import argparse

import torch

from . import curiosite
from .boucle import EtatSCL, boucle_temps_reel, cycle_nocturne
from .inne import construire_graphe_inne
from .logger import set_temps
from .memoires import TableBesoins
from .monde import Monde


def _inc_vision(graphe):
    m = graphe.modules.get("vision")
    return curiosite.incertitude(m) if m is not None else None


def evaluer(graine, n_jours, steps_par_jour, trace_par_jour=None):
    graphe, discriminateur = construire_graphe_inne()
    monde = Monde(graine=graine)
    etat = EtatSCL(graphe, discriminateur, monde, TableBesoins())

    inc_vision_j0 = None
    for jour in range(n_jours):
        set_temps(jour=jour)
        immobiles = 0
        for step in range(steps_par_jour):
            set_temps(step=step)
            v_av = (int(monde.vitesse[0]), int(monde.vitesse[1]))
            boucle_temps_reel(etat, t=step)
            if (int(monde.vitesse[0]), int(monde.vitesse[1])) == v_av == (0, 0):
                immobiles += 1
        cycle_nocturne(etat, t=steps_par_jour)
        if inc_vision_j0 is None:
            inc_vision_j0 = _inc_vision(graphe)
        if trace_par_jour is not None:
            rap = etat.dynamique.etat_maitrise()
            trace_par_jour.append(dict(
                jour=jour, inc_vision=round(_inc_vision(graphe), 4),
                immobile=round(immobiles / steps_par_jour, 2),
                n_pred=len(etat.dynamique.predicteurs),
                n_maitrises=sum(1 for _, (_, m) in rap.items() if m)))

    rap = etat.dynamique.etat_maitrise()
    return {
        "graine": graine,
        "inc_vision_debut": round(inc_vision_j0, 4) if inc_vision_j0 is not None else None,
        "inc_vision_fin": round(_inc_vision(graphe), 4),
        "n_predicteurs": len(etat.dynamique.predicteurs),
        "accels_apprises": sorted(list(a) for a in etat.dynamique.predicteurs),
        "n_maitrises": sum(1 for _, (_, m) in rap.items() if m),
        "detail_maitrise": rap,
        "sucres_incidents": monde.compteurs["sucre"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jours", type=int, default=15)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--graines", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--trace", action="store_true", help="trace jour par jour")
    args = p.parse_args()

    torch.manual_seed(0)
    for g in args.graines:
        trace = [] if args.trace else None
        r = evaluer(g, args.jours, args.steps, trace_par_jour=trace)
        print(f"[graine {g}] vision {r['inc_vision_debut']}→{r['inc_vision_fin']} | "
              f"prédicteurs dynamique={r['n_predicteurs']} "
              f"(accels {r['accels_apprises']}) | maîtrisées={r['n_maitrises']} | "
              f"sucres incidents={r['sucres_incidents']}")
        if trace:
            for d in trace:
                print(f"   j{d['jour']:2d}: inc_vision={d['inc_vision']:.4f} "
                      f"immobile={d['immobile']:.0%} n_pred={d['n_pred']} "
                      f"maîtrisées={d['n_maitrises']}")


if __name__ == "__main__":
    main()
