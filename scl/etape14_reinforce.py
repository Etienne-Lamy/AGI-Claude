"""ÉTAPE 14 — Mode B par RENFORCEMENT (§31.6) : découvrir, pas imiter.

Étape 13 montrait Mode B distillant les choix de Mode A par IMITATION. Ici on retire
le professeur : Mode B n'a QUE la récompense mesurée `R = G − λ·coût` (la même valeur
que Mode A utilise pour classer). Il ÉCHANTILLONNE un programme, encaisse R, et REINFORCE
remonte le gradient. On vérifie qu'il converge vers le meilleur programme PAR OBJECTIF —
la démonstration que l'orchestrateur-LLM apprend à émettre par renforcement selon le
contexte (Principe 2), sans qu'on lui donne la réponse.

    python3 -m scl.etape14_reinforce
"""
import argparse

import numpy as np

from .logger import set_temps
from .mode_b import ModeB, entrainer_par_renforcement
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

    print(f"Device : {DEVICE} — Mode A : mesure de R = G − λ·coût pour chaque programme…")
    classements = mode_A_multi(_flux_fn(v, 1), paires, objectifs=tuple(OBJECTIFS), pas=args.pas)

    # table de récompense : R(objectif, chaine) = valeur mesurée par Mode A.
    tables, gagnants = {}, {}
    for i, obj in enumerate(OBJECTIFS):
        tables[i] = {tuple(r["chaine"]): r["valeur"] for r in classements[obj]}
        gagnants[obj] = classements[obj][0]["chaine"]
        print(f"   objectif « {obj:<14} » → meilleur R : {gagnants[obj]} "
              f"(R={classements[obj][0]['valeur']:+.3f})")
    print("\n   table de récompense R(objectif, programme) :")
    for r in classements[OBJECTIFS[0]]:
        c = tuple(r["chaine"])
        print("      " + " → ".join(r["chaine"]).ljust(40)
              + "  ".join(f"{obj[:5]}={tables[i].get(c, float('nan')):+.3f}"
                          for i, obj in enumerate(OBJECTIFS)))

    def recompense(obj_idx, chaine):
        # programme non terminal / hors espace typé : forte pénalité.
        return tables[obj_idx].get(tuple(chaine), -1.0)

    print("\nMode B : REINFORCE depuis R seule (AUCUNE imitation, init aléatoire)…")
    mb = ModeB(n_objectifs=len(OBJECTIFS))
    entrainer_par_renforcement(mb, recompense, objectifs=range(len(OBJECTIFS)), pas=600)

    print("Mode B émet, après découverte par récompense :")
    ok = True
    for i, obj in enumerate(OBJECTIFS):
        emis = mb.emettre(i)
        conforme = emis == gagnants[obj]
        ok = ok and conforme
        print(f"   objectif « {obj:<14} » → {emis}  "
              f"{'✓' if conforme else '✗ (meilleur R : '+str(gagnants[obj])+')'}")
    print("\nOK — Mode B DÉCOUVRE le meilleur programme par objectif via la seule "
          "récompense R (§31.6, Principe 2)" if ok else
          "\nà affiner — REINFORCE n'a pas convergé vers l'optimum de R")


if __name__ == "__main__":
    main()
