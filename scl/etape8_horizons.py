"""ÉTAPE 8 — horizons T+1…T+H et BRANCHES d'action (§29.3, §30).

Une fois la règle N3 apprise (« (régime, action) → régime suivant »), prévoir plus
loin ne demande aucun mécanisme nouveau : c'est **exécuter le programme** — itérer
la règle sur une séquence d'actions. On mesure :

1. la **courbe G(h)** : gain de prédictibilité du régime à l'horizon h contre le
   prior trivial « le régime ne change pas ». Là où G(h) s'annule se trouve
   l'**horizon naturel** — au-delà, empiler ne sert à rien (§29.3) ;
2. les **branches** : depuis le MÊME état, deux séquences d'actions différentes
   donnent des trajectoires de régimes différentes — c'est la base de la
   planification (choisir la branche viendra quand un besoin donnera de la valeur).

    python3 -m scl.etape8_horizons --horizon 8
"""
import argparse
import random

import numpy as np

from .logger import set_temps
from .module_ae import DEVICE
from .pipeline import ACTIONS, construire


def _derouler(n3, regime0, actions_idx):
    """EXÉCUTE le programme : itère la règle N3 sur une séquence d'actions.
    Retourne la trajectoire de régimes prédits."""
    r, traj = regime0, []
    for a in actions_idx:
        r = n3.predire(r, a)
        traj.append(r)
    return traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--essais", type=int, default=250)
    p.add_argument("--pas_comp", type=int, default=2000)
    p.add_argument("--pas_regime", type=int, default=900)
    p.add_argument("--pas_action", type=int, default=2500)
    args = p.parse_args()

    print(f"Device : {DEVICE} — construction du pipeline…")
    comp, det, n3, idx, noms, m, assoc = construire(
        args.pas_comp, args.pas_regime, args.pas_action)
    H = args.horizon

    # --- évaluation multi-horizons : on déroule le programme et on compare au réel
    justes = [0] * H          # prédictions correctes à l'horizon h
    triviales = [0] * H       # prior « le régime ne change pas »
    totaux = [0] * H
    m.vitesse = np.array([0, 0], dtype=np.int64)
    det.delai.pousser(None)

    for essai in range(args.essais):
        set_temps(step=essai)
        # état de départ : régime courant
        r0 = None
        for _ in range(3):                      # quelques pas pour stabiliser l'identification
            r0, _, _ = det.identifier(m.percevoir()["vision"][-1]); m.appliquer_action((0, 0))
        if r0 is None:
            continue
        seq = [random.randrange(len(ACTIONS)) for _ in range(H)]
        pred = _derouler(n3, idx[r0], seq)      # ← exécution du programme
        # exécution RÉELLE de la même séquence
        reel = []
        for a in seq:
            m.appliquer_action(ACTIONS[a])
            ra, _, _ = det.identifier(m.percevoir()["vision"][-1])
            reel.append(idx[ra] if ra is not None else None)
        for h in range(H):
            if reel[h] is None:
                continue
            totaux[h] += 1
            justes[h] += int(pred[h] == reel[h])
            triviales[h] += int(idx[r0] == reel[h])

    print(f"\nCourbe G(h) — prévoir le régime à l'horizon h (sur {args.essais} essais) :")
    print(f"{'h':>3} {'exactitude':>11} {'trivial':>9} {'G(h)':>8}")
    horizon_naturel = 0
    for h in range(H):
        if not totaux[h]:
            continue
        ex = justes[h] / totaux[h]
        tr = triviales[h] / totaux[h]
        g = 1.0 - (1.0 - ex) / (1.0 - tr) if tr < 1.0 else 0.0
        if g > 0.05:
            horizon_naturel = h + 1
        print(f"{h+1:>3} {ex:>10.0%} {tr:>9.0%} {g:>+8.0%}")
    print(f"\nHorizon naturel (dernier h avec G>5%) : **T+{horizon_naturel}**")

    # --- branches : même état, deux programmes différents → futurs différents
    r0, _, _ = det.identifier(m.percevoir()["vision"][-1])
    if r0 is not None:
        a_tout_droit = [ACTIONS.index((0, 0))] * H
        a_accelere = [ACTIONS.index((1, 0))] * H
        t1 = _derouler(n3, idx[r0], a_tout_droit)
        t2 = _derouler(n3, idx[r0], a_accelere)
        print(f"\nBranches depuis le régime « {noms[idx[r0]]} » :")
        print(f"   actions (0,0)×{H} → " + " → ".join(noms[r] for r in t1))
        print(f"   actions (1,0)×{H} → " + " → ".join(noms[r] for r in t2))
        print("   (deux programmes, deux futurs imaginés : c'est la base des branches §29.3)")


if __name__ == "__main__":
    main()
