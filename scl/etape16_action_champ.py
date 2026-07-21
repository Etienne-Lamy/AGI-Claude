"""ÉTAPE 16 — la conséquence d'une ACTION sur le champ est prévisible (§10 conception).

Un module champ→champ par action apprend l'effet de cette action MAINTENUE : l'accélération
soutenue fait saturer la vitesse (±v_max), induisant un FLUX de champ fort et distinct par
action (c'est le régime de vitesse de l'étape 10, mais désormais indexé par l'action que
l'agent ÉMET — copie d'efférence). Attendu : la **matrice croisée** de rappel est DIAGONALE
— le module d'une action prédit bien SON flux et mal celui des autres.

Leçon d'une première version (vitesse remise à 0) : l'effet d'UN pas d'accélération = 1
cellule de décalage, trop faible/proche entre actions pour être séparable sur un champ
épars (matrice plate ~50 %). La conséquence UTILE d'une action est son effet SOUTENU — ce
que la navigation exploitera (prévoir plusieurs pas d'une même commande).

    python3 -m scl.etape16_action_champ
"""
import argparse

import numpy as np

from .action import TransitionActionChamp
from .logger import set_temps
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde


def _segment(m, a, n, tac=None, collect=None, rampe=4):
    """Applique l'action `a` de façon SOUTENUE sur n pas (vitesse rampe puis sature).
    Entraîne le module de `a` (si tac) et/ou collecte les transitions (si collect).
    On saute les `rampe` premiers pas (vitesse pas encore saturée)."""
    m.vitesse = np.array([0, 0], dtype=np.int64)
    prec = None
    for s in range(n):
        champ = np.asarray(m.percevoir()["vision"][-1]).copy()
        if prec is not None and s >= rampe:
            if tac is not None:
                tac.observer(prec, a, champ)
            if collect is not None:
                collect.append((prec, a, champ))
        prec = champ
        m.appliquer_action(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=800)       # pas par action
    p.add_argument("--eval", type=int, default=120)
    args = p.parse_args()
    actions = ACCELERATIONS_PERMISES
    print(f"Device : {DEVICE} — {len(actions)} actions, {args.pas} pas/action (soutenue)")

    tac = TransitionActionChamp(actions)
    for a in actions:                       # un module à la fois (watchdog)
        m = Monde(graine=1)
        _segment(m, a, args.pas, tac=tac)
        print(f"   action {str(a):>7} : module entraîné ({tac.n_maj[a]} maj)", flush=True)

    # held-out : nouvelles transitions par action, sans apprentissage
    transitions = []
    for a in actions:
        me = Monde(graine=99)
        _segment(me, a, args.eval, collect=transitions)
    mat = tac.matrice_rappel(transitions)

    print("\nMatrice croisée — rappel du module LIGNE sur le flux de l'action COLONNE :")
    print("            " + "  ".join(f"{str(b):>7}" for b in actions))
    diag, off = [], []
    for a in actions:
        print(f"{str(a):>11} " + "  ".join(f"{mat[a][b]:7.0%}" for b in actions))
        for b in actions:
            (diag if a == b else off).append(mat[a][b])
    dm, om = float(np.mean(diag)), float(np.mean(off))
    print(f"\nDiagonale (bonne action) : {dm:.0%}  vs  hors-diagonale : {om:.0%}  "
          f"→ écart {dm - om:+.0%}")
    ok = dm > om + 0.10
    print("\nOK — l'effet SOUTENU de l'action sur le champ est prévisible ET séparable"
          if ok else "\nà affiner — la matrice n'est pas assez diagonale")


if __name__ == "__main__":
    main()
