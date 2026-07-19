"""ÉTAPE 7 — hiérarchie N2→N3 : l'ACCÉLÉRATION émerge comme règle apprise.

Le pipeline (compresseur gelé → vocabulaire de régimes verrouillé → règle N3) est
construit par `scl.pipeline`. Cette étape mesure ce qu'il démontre :

- **N3** prédit le régime suivant à partir de `(régime, action)` — c'est
  l'accélération, apprise comme un **modèle de transition sur l'espace des modules**,
  jamais câblée ni lue par un capteur ;
- son **gain** est mesuré contre le prior trivial « le régime ne change pas » ;
- la **règle apprise est imprimée**, donc lisible : quelle action fait passer de v1 à v2.

    python3 -m scl.etape7_hierarchie
"""
import argparse
import random

import numpy as np

from .hierarchie import gain_vs_trivial
from .logger import set_temps
from .module_ae import DEVICE
from .pipeline import ACTIONS, construire


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas_comp", type=int, default=2000)
    p.add_argument("--pas_regime", type=int, default=900)
    p.add_argument("--pas_action", type=int, default=2500)
    p.add_argument("--pas_eval", type=int, default=800)
    args = p.parse_args()

    print(f"Device : {DEVICE} — construction du pipeline…")
    comp, det, n3, idx, noms, m, assoc = construire(
        args.pas_comp, args.pas_regime, args.pas_action)
    print(f"\nVocabulaire N2 : {len(idx)} régimes → {noms}")
    for v, vid in assoc.items():
        print(f"   vitesse {str(v):>8} → régime « {vid} »")

    # --- évaluation de N3 (aucun apprentissage ici)
    m.vitesse = np.array([0, 0], dtype=np.int64)
    det.delai.pousser(None)
    prec, preds, verites, precedents = None, [], [], []
    for s in range(args.pas_eval):
        set_temps(step=s)
        a = random.choice(ACTIONS)
        actif, _, _ = det.identifier(m.percevoir()["vision"][-1])
        m.appliquer_action(a)
        if prec is not None and actif is not None:
            r_prec, i_a = idx[prec[0]], ACTIONS.index(prec[1])
            preds.append(n3.predire(r_prec, i_a))
            verites.append(idx[actif]); precedents.append(r_prec)
        prec = (actif, a) if actif is not None else None

    just, triv, gain, n_chg = gain_vs_trivial(preds, verites, precedents)
    print(f"\nN3 — prédire le régime suivant depuis (régime, action) :")
    print(f"   exactitude N3   : {just:.0%}")
    print(f"   prior trivial   : {triv:.0%}   (« le régime ne change pas »)")
    print(f"   GAIN vs trivial : {gain:+.0%}   ({n_chg} changements réels / {len(verites)} pas)")

    print("\nRÈGLE APPRISE (régime, action) → régime suivant :")
    table = n3.table(noms_regimes=noms, noms_actions=[str(x) for x in ACTIONS])
    for (r, a), suiv in table.items():
        print(f"   {str(r):>8} + accel {a:>8} → {str(suiv):>8}" + ("  ← change" if r != suiv else ""))
    print("\nOK — l'effet d'une action sur le régime est APPRIS" if gain > 0.1
          else "\nà affiner — gain faible sur le prior trivial")


if __name__ == "__main__":
    main()
