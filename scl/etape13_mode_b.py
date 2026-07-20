"""ÉTAPE 13 — Mode A → Mode B (§31.6) : distiller la recherche en émission apprise.

1. Mode A entraîne les programmes UNE fois et les classe pour DEUX objectifs :
   « prédire le champ suivant » et « reconstruire le champ courant ». Les meilleurs
   programmes DIFFÈRENT (predire_champ vs compresser→generer) — le contexte compte.
2. On DISTILLE ces choix dans Mode B (imitation) : Mode B émet ensuite le bon
   programme par objectif SANS refaire la recherche (il amortit le coût de A).

    python3 -m scl.etape13_mode_b
"""
import argparse

import numpy as np

from .logger import set_temps
from .mode_b import ModeB, entrainer_par_imitation
from .module_ae import DEVICE
from .monde import Monde
from .orchestrateur import mode_A_multi

OBJECTIFS = ["prediction", "reconstruction"]


def _flux_fn(vitesse, graine):
    def gen(pas):
        m = Monde(graine=graine); m.vitesse = np.array(vitesse, dtype=np.int64)
        prec = None
        for s in range(pas):
            set_temps(step=s)
            champ = np.asarray(m.percevoir()["vision"][-1]).copy()
            yield prec, champ
            prec = champ
            m.appliquer_action((0, 0))
    return gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=1200)
    p.add_argument("--vitesse", type=int, nargs=2, default=[1, 1])
    args = p.parse_args()
    v = tuple(args.vitesse)

    me = Monde(graine=555); me.vitesse = np.array(v, dtype=np.int64)
    paires, prec = [], None
    for _ in range(80):
        champ = np.asarray(me.percevoir()["vision"][-1]).copy()
        if prec is not None:
            paires.append((prec, champ))
        prec = champ
        me.appliquer_action((0, 0))

    print(f"Device : {DEVICE} — Mode A : entraînement + classement pour 2 objectifs…")
    classements = mode_A_multi(_flux_fn(v, 1), paires, objectifs=tuple(OBJECTIFS), pas=args.pas)

    exemples, gagnants = [], {}
    for i, obj in enumerate(OBJECTIFS):
        best = classements[obj][0]
        gagnants[obj] = best["chaine"]
        exemples.append((i, best["chaine"]))
        print(f"   objectif « {obj:<14} » → Mode A choisit {best['chaine']} (G={best['G']:.0%})")

    print("\nMode B : distillation par imitation des choix de Mode A…")
    mb = ModeB(n_objectifs=len(OBJECTIFS))
    entrainer_par_imitation(mb, exemples, pas=500)

    print("Mode B émet (sans aucune recherche) :")
    ok = True
    for i, obj in enumerate(OBJECTIFS):
        emis = mb.emettre(i)
        conforme = emis == gagnants[obj]
        ok = ok and conforme
        print(f"   objectif « {obj:<14} » → {emis}  {'✓' if conforme else '✗ (attendu '+str(gagnants[obj])+')'}")
    print("\nOK — Mode B reproduit les choix de Mode A par objectif, SANS recherche (§31.6)"
          if ok else "\nà affiner — Mode B ne reproduit pas tous les choix")


if __name__ == "__main__":
    main()
