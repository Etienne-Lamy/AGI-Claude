"""ÉTAPE 21 — la CURIOSITÉ stoppe le vol rectiligne (point de l'auteur).

Quand plus rien ne s'apprend en exploitant ce qu'on sait (le vol rectiligne de l'étape 20),
l'ennui doit pousser à sortir de la zone de confort : LISTER les actions et viser celle dont
on prévoit le MOINS bien la conséquence (incertitude du modèle de transition, étape 16).

On oppose deux politiques qui apprennent le même modèle en ligne :
  - CURIEUSE  : argmax d'incertitude → essaie ce qu'elle ne sait pas prévoir ;
  - EXPLOITANTE : argmin d'incertitude → refait ce qu'elle prévoit déjà (→ s'effondre).
Attendu : la curieuse garde une entropie d'action haute ET fait BAISSER l'incertitude de
TOUTES les actions (les maîtrise « de proche en proche »), là où l'exploitante se fige.

    python3 -m scl.etape21_curiosite
"""
import argparse

import numpy as np

from .action import TransitionActionChamp
from .logger import set_temps
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde
from .planification import choisir_curieux, incertitude_module


def _run(sens, pas, graine):
    """sens=+1 : curieuse (argmax incertitude) ; sens=-1 : exploitante (argmin)."""
    actions = ACCELERATIONS_PERMISES
    tac = TransitionActionChamp(actions)
    m = Monde(graine=graine)
    compte = np.zeros(len(actions))
    prec = None
    for s in range(pas):
        set_temps(step=s)
        champ = np.asarray(m.percevoir()["vision"][-1]).copy()
        if sens > 0:
            a_idx = choisir_curieux(tac, actions, epsilon=0.0)
        else:
            inc = [incertitude_module(tac.modules[a]) for a in actions]
            a_idx = int(np.argmin(inc))
        if prec is not None:
            tac.observer(prec, actions[a_idx], champ)   # apprend la conséquence vécue
        compte[a_idx] += 1
        prec = champ
        m.appliquer_action(actions[a_idx])
    inc_fin = [incertitude_module(tac.modules[a]) for a in actions]
    p = compte / compte.sum()
    entropie = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(actions)))
    return entropie, inc_fin, compte


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=2500)
    args = p.parse_args()
    actions = ACCELERATIONS_PERMISES
    print(f"Device : {DEVICE} — curiosité vs exploitation, {args.pas} pas")

    ec, incc, cptc = _run(+1, args.pas, graine=5)
    ee, ince, cpte = _run(-1, args.pas, graine=5)

    def ligne(nom, ent, inc, cpt):
        print(f"\n{nom} : entropie d'action {ent:.2f} (1 = toutes égales)")
        print("   répartition : " + "  ".join(f"{str(a)}:{int(c)}" for a, c in zip(actions, cpt)))
        print("   incertitude finale/action : " + "  ".join(f"{v:.2f}" for v in inc))
    ligne("CURIEUSE", ec, incc, cptc)
    ligne("EXPLOITANTE", ee, ince, cpte)

    seuil = 0.5
    maitr_c = sum(1 for v in incc if v < seuil)
    maitr_e = sum(1 for v in ince if v < seuil)
    print(f"\nActions maîtrisées (incertitude < {seuil}) : curieuse {maitr_c}/{len(actions)} "
          f"vs exploitante {maitr_e}/{len(actions)}")
    print(f"Entropie d'action : curieuse {ec:.2f} vs exploitante {ee:.2f}")
    ok = ec > ee + 0.15 and maitr_c >= maitr_e
    print("\nOK — la curiosité garde la diversité d'action et maîtrise plus d'actions : "
          "elle sort du vol rectiligne, là où l'exploitation se fige" if ok else
          "\nà affiner — la curiosité ne se distingue pas assez de l'exploitation")


if __name__ == "__main__":
    main()
