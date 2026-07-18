"""ÉTAPE 2a — prédiction du champ suivant, à la main (avant l'orchestrateur).

Deux modules prédictifs bâtis à la main sur l'objet générique `ModuleAutoencodeur` :
  - P_visuel   : champ visuel  P-1 → champ visuel  P
  - P_abstrait : champ abstrait P-1 → champ abstrait P (via le module 1, vision)

CLAIM À VALIDER (fondements, §12/§4) : à VITESSE FIXE, le champ présent est le
champ précédent DÉCALÉ de la vitesse. Un prédicteur entraîné à une vitesse v₁
prédit bien à v₁ (fiabilité haute) et mal aux autres vitesses (le décalage
diffère). Donc la FIABILITÉ D'UN PRÉDICTEUR EST UN INDICATEUR DE VITESSE — c'est
exactement ce que l'orchestrateur pourra lire et prédire.

Usage : python3 -m scl.etape2_prediction --pas 4000
"""
import argparse

import numpy as np

from .config import CONFIG
from .logger import set_temps
from .module_ae import DEVICE, ModuleAutoencodeur
from .monde import Monde

VITESSES_TEST = [(1, 1), (2, 0), (0, 1), (1, -1), (2, 2)]


def _flux(graine, vitesse, n):
    """Génère n frames consécutives à vitesse fixe (retour : liste de champs)."""
    m = Monde(graine=graine)
    m.vitesse = np.array(vitesse, dtype=np.int64)
    frames = []
    for _ in range(n):
        frames.append(np.asarray(m.percevoir()["vision"][-1]).copy())
        m.appliquer_action((0, 0))
    return frames


def _fiabilite_a_vitesse(pred, vitesse, graine=7, n=60):
    """Fiabilité (exactitude/rappel de prédiction P-1→P) sur un flux FRAIS à
    cette vitesse, SANS entraîner."""
    frames = _flux(graine, vitesse, n + 1)
    ex, ra = [], []
    for i in range(n):
        d = pred.fidelite_transition(frames[i], frames[i + 1])
        ex.append(d["exactitude"]); ra.append(d["rappel"])
    return sum(ex) / len(ex), sum(ra) / len(ra)


def valider(n_pas=2500, vitesse_train=(1, 1), graine=1):
    """Entraîne le prédicteur visuel P-1→P à vitesse fixe (le seul module requis
    pour la revendication « fiabilité = indicateur de vitesse »)."""
    p_vis = ModuleAutoencodeur("pred_visuel")
    monde = Monde(graine=graine)
    monde.vitesse = np.array(vitesse_train, dtype=np.int64)
    champ_prec = None
    for pas in range(n_pas):
        set_temps(step=pas)
        champ = monde.percevoir()["vision"][-1]
        if champ_prec is not None:
            p_vis.entrainer_transition(champ_prec, champ)
        champ_prec = champ
        monde.appliquer_action((0, 0))
    return p_vis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=4000)
    p.add_argument("--train", type=int, nargs=2, default=[1, 1])
    args = p.parse_args()

    print(f"Device : {DEVICE}")
    vt = tuple(args.train)
    p_vis = valider(args.pas, vitesse_train=vt)

    print(f"\nPrédicteur visuel ENTRAÎNÉ à la vitesse {vt}. Fiabilité par vitesse :")
    print(f"{'vitesse':>10} {'exactitude':>11} {'rappel_pred':>12} {'':>4}")
    ref, autres = None, []
    for v in VITESSES_TEST:
        ex, ra = _fiabilite_a_vitesse(p_vis, v)
        if v == vt:
            ref = ra
        else:
            autres.append(ra)
        print(f"{str(v):>10} {ex:11.1%} {ra:12.1%}{'  ← ENTRAÎNÉE' if v == vt else ''}")
    # le rappel de prédiction (pas l'exactitude, dominée par le vide) discrimine
    ok = ref is not None and ref > max(autres) + 0.2
    print(f"\nVERDICT étape 2a : rappel de prédiction à {vt} = {ref:.0%} vs max "
          f"ailleurs = {max(autres):.0%} → "
          f"{'OK — la fiabilité EST un indicateur de vitesse' if ok else 'à revoir'}")


if __name__ == "__main__":
    main()
