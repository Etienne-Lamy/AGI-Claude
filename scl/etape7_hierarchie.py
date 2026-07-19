"""ÉTAPE 7 — hiérarchie N2→N3 : l'ACCÉLÉRATION émerge comme règle apprise.

Phase 1 (vocabulaire) : on parcourt des vitesses constantes ; un module-vitesse
NAÎT par régime et se VERROUILLE (étape 6). On obtient le signal **N2** = « quel
module explique la transition ».

Phase 2 (règle) : l'agent AGIT (accélérations). À chaque pas on identifie le régime
actif, et un module **N3** apprend `(régime, action) → régime suivant`. On mesure
son gain contre le prior trivial « le régime ne change pas », et on imprime la
RÈGLE APPRISE — c'est-à-dire quelle action fait passer de v1 à v2.

    python3 -m scl.etape7_hierarchie
"""
import argparse
import random

import numpy as np

from .composition import DetecteurVitesse
from .config import CONFIG
from .hierarchie import ModuleTransitionRegime, gain_vs_trivial
from .logger import set_temps
from .module_ae import DEVICE, ModuleAutoencodeur
from .monde import Monde

VITESSES = [(0, 0), (1, 0), (2, 0), (-1, 0), (-2, 0)]      # 1D : 5 régimes
ACTIONS = [(0, 0), (1, 0), (-1, 0)]                         # 3 accélérations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas_comp", type=int, default=2000)
    p.add_argument("--pas_regime", type=int, default=900)
    p.add_argument("--pas_action", type=int, default=2500)
    args = p.parse_args()

    # ---- compresseur (module 1), puis GELÉ
    comp = ModuleAutoencodeur("compresseur")
    m = Monde(graine=1); m.vitesse = np.array([1, 0], dtype=np.int64)
    print(f"Device : {DEVICE} — compresseur…")
    for s in range(args.pas_comp):
        set_temps(step=s); comp.entrainer(m.percevoir()["vision"][-1]); m.appliquer_action((0, 0))

    # ---- PHASE 1 : vocabulaire de régimes (un module-vitesse par vitesse)
    det = DetecteurVitesse(comp)
    module_de_vitesse = {}
    print("\nPhase 1 — formation du vocabulaire de régimes :")
    for v in VITESSES:
        det.delai.pousser(None)
        m.vitesse = np.array(v, dtype=np.int64)
        avant = set(det.vitesses)
        for s in range(args.pas_regime):
            set_temps(step=s); det.etape(m.percevoir()["vision"][-1]); m.appliquer_action((0, 0))
        nes = set(det.vitesses) - avant
        # le module associé à cette vitesse = celui qui l'explique le mieux à la fin
        det.delai.pousser(None)
        scores = {}
        for _ in range(60):
            actif, res, _ = det.identifier(m.percevoir()["vision"][-1]); m.appliquer_action((0, 0))
            if actif: scores[actif] = scores.get(actif, 0) + 1
        assoc = max(scores, key=scores.get) if scores else None
        module_de_vitesse[v] = assoc
        print(f"   v={str(v):>8} : {len(det.vitesses)} modules (+{len(nes)}) → régime associé « {assoc} »")
    for mv in det.vitesses.values():          # tous spécialistes : on fige
        mv.verrouille = True

    ids = sorted(det.vitesses)
    idx = {vid: i for i, vid in enumerate(ids)}
    print(f"\nVocabulaire N2 : {len(ids)} régimes {ids}")

    # ---- PHASE 2 : l'agent AGIT ; N3 apprend (régime, action) → régime suivant
    n3 = ModuleTransitionRegime("n3_transition", len(ids), len(ACTIONS))
    m.vitesse = np.array([0, 0], dtype=np.int64)
    det.delai.pousser(None)
    prec, preds, verites, precedents = None, [], [], []
    print("\nPhase 2 — l'agent agit ; N3 apprend la règle action→régime…")
    for s in range(args.pas_action):
        set_temps(step=s)
        a = random.choice(ACTIONS)
        actif, _, _ = det.identifier(m.percevoir()["vision"][-1])
        m.appliquer_action(a)
        if prec is not None and actif is not None:
            r_prec, r_act, i_a = idx[prec[0]], idx[actif], ACTIONS.index(prec[1])
            if s > args.pas_action * 0.7:            # dernier tiers = évaluation
                preds.append(n3.predire(r_prec, i_a))
                verites.append(r_act); precedents.append(r_prec)
            else:
                n3.entrainer(r_prec, i_a, r_act)
        prec = (actif, a) if actif is not None else None

    just, triv, gain, n_chg = gain_vs_trivial(preds, verites, precedents)
    print(f"\nN3 — prédiction du régime suivant à partir de (régime, action) :")
    print(f"   exactitude N3      : {just:.0%}")
    print(f"   prior trivial      : {triv:.0%}   (« le régime ne change pas »)")
    print(f"   GAIN vs trivial    : {gain:+.0%}   ({n_chg} changements réels sur {len(verites)} pas)")

    inv = {v: k for k, v in module_de_vitesse.items() if v}
    noms = [f"{inv.get(vid, '?')}" for vid in ids]
    print("\nRÈGLE APPRISE (régime, action) → régime suivant :")
    for (r, a), suiv in n3.table(noms_regimes=noms, noms_actions=[str(x) for x in ACTIONS]).items():
        marque = "  ← change" if r != suiv else ""
        print(f"   {str(r):>8} + accel {a:>8} → {str(suiv):>8}{marque}")
    print("\nOK — l'effet d'une action sur le régime est APPRIS" if gain > 0.1
          else "\nà affiner — le gain sur le prior trivial reste faible")


if __name__ == "__main__":
    main()
