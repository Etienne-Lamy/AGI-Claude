"""ÉTAPE 11 — Orchestrateur Mode A : DÉCOUVRIR la meilleure composition par mesure.

L'orchestrateur énumère les programmes typés qui prédisent le champ suivant, entraîne
chacun, mesure son gain de prédictibilité G et son coût, et classe par valeur. On
vérifie qu'il RETROUVE SEUL (sans préférence câblée) que la prédiction en espace-champ
bat la chaîne en latent compressé opaque — c.-à-d. qu'il redécouvre STATUS §5bis.

    python3 -m scl.etape11_orchestrateur
"""
import argparse
import numpy as np

from .logger import set_temps
from .module_ae import DEVICE
from .monde import Monde
from .orchestrateur import mode_A


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
    p.add_argument("--pas", type=int, default=1500)
    p.add_argument("--vitesse", type=int, nargs=2, default=[1, 1])
    args = p.parse_args()
    v = tuple(args.vitesse)

    # jeu d'évaluation tenu à l'écart
    me = Monde(graine=555); me.vitesse = np.array(v, dtype=np.int64)
    paires, prec = [], None
    for _ in range(80):
        champ = np.asarray(me.percevoir()["vision"][-1]).copy()
        if prec is not None:
            paires.append((prec, champ))
        prec = champ
        me.appliquer_action((0, 0))

    print(f"Device : {DEVICE} — Mode A énumère et mesure les programmes typés…")
    res = mode_A(_flux_fn(v, 1), paires, profondeur_max=3, pas=args.pas)

    print(f"\n{'programme (chaîne d’opérateurs)':<48}{'G':>7}{'coût':>6}{'valeur':>8}")
    for r in res:
        etoile = "  ← CHOISI" if r is res[0] else ""
        print(f"{' → '.join(r['chaine']):<48}{r['G']:>7.0%}{r['cout']:>6}{r['valeur']:>8.2f}{etoile}")
    best = res[0]["chaine"]
    champ_direct = ["predire_champ"]
    print(f"\nProgramme choisi : {' → '.join(best)}")
    print("OK — l'orchestrateur a redécouvert que l'espace-champ prédit le mieux (§5bis)"
          if best == champ_direct else
          "à analyser — le programme choisi n'est pas la prédiction champ-directe")


if __name__ == "__main__":
    main()
