"""ÉTAPE 10 — détection de régime en ESPACE-CHAMP (lève le goulot §5bis).

Refait les étapes 6 (détection de vitesse) et 9 (vent), mais les modules-régime
prédisent champ→champ (rappel objets ∈ [0,1]) au lieu du latent opaque. Attendu :

- **détection nette** : chaque module a un rappel élevé sur SON régime, bas ailleurs ;
- **vent transverse détecté** : un vent en Y sort du vocabulaire 1D-en-x → le rappel
  s'effondre → un module NAÎT (ce que le latent opaque ne parvenait pas à faire).

    python3 -m scl.etape10_regime_champ
"""
import argparse

import numpy as np

from .logger import set_temps
from .module_ae import DEVICE
from .monde import Monde
from .regime import DetecteurRegimeChamp

VITESSES = [(1, 0), (2, 0), (0, 1)]


def _defiler(det, monde, v, n, apprendre=True):
    monde.vitesse = np.array(v, dtype=np.int64)
    det.champ_prec = None
    avant = len(det.regimes)
    for s in range(n):
        set_temps(step=s)
        det.etape(monde.percevoir()["vision"][-1], apprendre=apprendre)
        monde.appliquer_action((0, 0))
    return len(det.regimes) - avant


def _matrice(det, graine=321, n=40):
    m = Monde(graine=graine)
    lignes = {mid: {} for mid in det.regimes}
    for v in VITESSES:
        m.vitesse = np.array(v, dtype=np.int64)
        det.champ_prec = None
        acc = {mid: [] for mid in det.regimes}
        for _ in range(n):
            champ = m.percevoir()["vision"][-1]
            if det.champ_prec is not None:
                for mid, r in det.rappels(det.champ_prec, champ).items():
                    acc[mid].append(r)
            det.champ_prec = champ
            m.appliquer_action((0, 0))
        for mid in det.regimes:
            lignes[mid][v] = float(np.mean(acc[mid])) if acc[mid] else 0.0
    return lignes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas_regime", type=int, default=3000)
    p.add_argument("--vent", type=int, nargs=2, default=[0, 2])
    args = p.parse_args()
    print(f"Device : {DEVICE}")

    det = DetecteurRegimeChamp()
    m = Monde(graine=1)
    print("Formation du vocabulaire de régimes (espace-champ) :")
    for v in VITESSES:
        nes = _defiler(det, m, v, args.pas_regime)
        print(f"   v={str(v):>7} : {len(det.regimes)} modules (+{nes})", flush=True)
    det.verrouiller_tout()

    print("\nMatrice de détection — RAPPEL objets (haut = ce module prédit ce régime) :")
    print("            " + "  ".join(f"{str(v):>8}" for v in VITESSES))
    lignes = _matrice(det)
    couvertes = set()
    for mid, l in lignes.items():
        gagne = max(VITESSES, key=lambda v: l[v])
        couvertes.add(gagne)
        print(f"{mid:>11} " + "  ".join(f"{l[v]:8.0%}" for v in VITESSES) + f"   → {gagne}")
    print(f"Vitesses distinctes couvertes : {len(couvertes)}/{len(VITESSES)}")

    # --- VENT : régime hors vocabulaire → familiarité chute → un module naît
    print(f"\nVENT {tuple(args.vent)} (hors vocabulaire) :")
    m2 = Monde(graine=7); m2.vent = np.array(args.vent, dtype=np.int64)
    m2.vitesse = np.array([1, 0], dtype=np.int64)
    det.champ_prec = None
    fam = []
    for _ in range(120):
        champ = m2.percevoir()["vision"][-1]
        _, _, f = det.identifier(champ)
        if det.champ_prec is not None or f:
            fam.append(f)
        m2.appliquer_action((0, 0))
    fam_vent = float(np.mean([x for x in fam if x])) if any(fam) else 0.0
    fam_ref = max(max(l.values()) for l in lignes.values())
    print(f"   familiarité sous vent : {fam_vent:.0%}  (vs {fam_ref:.0%} sur régime connu)")
    avant = len(det.regimes)
    _defiler(det, m2, [1, 0], 1200)          # adaptation libre (vent toujours actif via m2.vent)
    nes = len(det.regimes) - avant
    print(f"   modules nés sous vent : {nes}")
    ok = len(couvertes) >= 2 and nes >= 1
    print("\nOK — détection nette ET vent reconnu comme régime nouveau" if ok
          else "\nà affiner")


if __name__ == "__main__":
    main()
