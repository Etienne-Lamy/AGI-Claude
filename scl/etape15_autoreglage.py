"""ÉTAPE 15 — auto-réglage §28.4 BRANCHÉ sur l'orchestrateur.

On applique la boucle réversible (§28.4) à un vrai levier : `grace_regime`, contre le
résidu MESURÉ de sur-création de modules-régime (étape 10 : 5 modules pour 3 régimes).
Observable à MAXIMISER = couverture des régimes − pénalité de parcimonie (modules au-delà
de l'idéal). L'auto-régleur essaie des valeurs de grâce, GARDE celle qui améliore
l'observable et REVIENT en arrière sinon — sans qu'on lui dise la bonne valeur.

    python3 -m scl.etape15_autoreglage
"""
import argparse

import numpy as np

from .autoreglage import AutoReglage
from .config import CONFIG
from .etape10_regime_champ import VITESSES, _defiler, _matrice
from .module_ae import DEVICE
from .monde import Monde
from .regime import DetecteurRegimeChamp

LAMBDA_PARCIMONIE = 0.30      # poids de la pénalité « modules au-delà de l'idéal »


def _mesurer(grace, pas_regime):
    """Construit un détecteur avec cette grâce, l'entraîne sur les régimes, et rend
    l'observable = couverture − λ·(modules superflus). Plus grand = mieux."""
    CONFIG["grace_regime"] = int(grace)
    det = DetecteurRegimeChamp()
    m = Monde(graine=1)
    for v in VITESSES:
        _defiler(det, m, v, pas_regime)
    det.verrouiller_tout()
    lignes = _matrice(det)                       # rappel de chaque module par régime
    couverture = float(np.mean([max(lignes[mid][v] for mid in det.regimes)
                                for v in VITESSES]))      # ∈[0,1] : chaque régime bien prédit ?
    superflus = max(0, len(det.regimes) - len(VITESSES))
    obs = couverture - LAMBDA_PARCIMONIE * superflus / len(VITESSES)
    print(f"   grace={int(grace):>5} → {len(det.regimes)} modules, "
          f"couverture={couverture:.0%}, observable={obs:+.3f}", flush=True)
    return obs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas_regime", type=int, default=1000)
    args = p.parse_args()
    print(f"Device : {DEVICE} — auto-réglage §28.4 de grace_regime (réversible)\n")

    base = CONFIG["grace_regime"]
    reg = AutoReglage()
    _, score = reg.regler(
        "grace_regime", base, deltas=(-1000, +2000),
        appliquer=lambda v: CONFIG.__setitem__("grace_regime", int(v)),
        mesurer=lambda: _mesurer(CONFIG["grace_regime"], args.pas_regime),
    )

    print(f"\nBilan §28.4 : grace_regime {base} → {reg.historique[-1]['apres']} "
          f"({'GARDÉ' if reg.historique[-1]['garde'] else 'REVERT — la base restait la meilleure'}),"
          f" observable={score:+.3f}")
    CONFIG["grace_regime"] = base                # on ne persiste pas un réglage global ici
    print("OK — la boucle réversible a réglé un vrai levier de l'orchestrateur par la "
          "seule mesure (§28.4)")


if __name__ == "__main__":
    main()
