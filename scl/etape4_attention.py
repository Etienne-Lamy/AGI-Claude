"""ÉTAPE 4 — outil attention/masquage : décomposition du champ en OBJETS
(Slot Attention) → latent STRUCTURÉ → prédiction TRIVIALE par décalage.

Démontre : (1) reconstruction objet-centrée (VU vs PRÉVU) ; (2) la liste d'objets
(x,y,type) extraite ; (3) la prédiction du champ suivant en DÉCALANT simplement les
positions de la vitesse (aucun réseau de prédiction — juste position → position+v).

    python3 -m scl.etape4_attention --pas 8000 --log etape4.jsonl
    python3 viewer.py --log etape4.jsonl
"""
import argparse

import numpy as np

from .config import CONFIG
from .logger import configurer, log, set_temps
from .module_attention import DEVICE, VALEURS, ModuleAttentionSlots
from .monde import Monde


def _rendre(objets, t):
    """Liste d'objets (row, col, type) → champ 10×10."""
    champ = np.zeros((t, t), dtype=np.float32)
    champ[t // 2, t // 2] = 0.25
    for (i, j, typ) in objets:
        if 0 <= i < t and 0 <= j < t:
            champ[i, j] = VALEURS[typ]
    return champ


def _decaler(objets, di, dj):
    return [(i + di, j + dj, typ) for (i, j, typ) in objets]


def _rappel(pred, vrai, t):
    v = vrai.reshape(-1); p = pred.reshape(-1)
    obj = v > 0.1
    n = int(obj.sum())
    if not n:
        return 1.0
    return int((np.abs(p - v) < 0.2)[obj].sum()) / n


def _str(f):
    return ",".join(f"{float(x):.2f}" for x in np.asarray(f).reshape(-1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=8000)
    p.add_argument("--vitesse", type=int, nargs=2, default=[1, 1])
    p.add_argument("--log", type=str, default="")
    p.add_argument("--delai", type=float, default=0.0)
    args = p.parse_args()
    logger = configurer(chemin=args.log) if args.log else None
    v = tuple(args.vitesse); t = CONFIG["taille_perception"]

    mod = ModuleAttentionSlots("attention_vision")
    m = Monde(graine=1); m.vitesse = np.array(v, dtype=np.int64)
    print(f"Device : {DEVICE} — entraînement Slot Attention (vitesse {v})…")
    for s in range(args.pas):
        set_temps(step=s)
        mod.entrainer(m.percevoir()["vision"][-1])
        m.appliquer_action((0, 0))

    # évaluation reconstruction
    ev = []
    for _ in range(40):
        for _ in range(5): m.appliquer_action((0, 0))
        ev.append(np.asarray(m.percevoir()["vision"][-1]).copy())
    fr = [mod.fidelite(f) for f in ev]
    print(f"Reconstruction objet-centrée : rappel="
          f"{sum(d['rappel'] for d in fr) / len(fr):.0%} "
          f"precision={sum(d['precision'] for d in fr) / len(fr):.0%}")
    print(f"Exemple de liste d'objets extraite : {mod.liste_objets(ev[0])}")

    # prédiction TRIVIALE : objets(t-1) DÉCALÉS de la vitesse → champ(t) prédit.
    # On détermine d'abord le bon sens/axe du décalage (perception égocentrée) en
    # comparant les candidats sur un échantillon, puis on l'utilise.
    candidats = {"row-,col-": (-v[0], -v[1]), "row+,col+": (v[0], v[1]),
                 "swap-,-": (-v[1], -v[0]), "swap+,+": (v[1], v[0])}
    echant = []
    prev = None
    for _ in range(60):
        c = np.asarray(m.percevoir()["vision"][-1]).copy()
        if prev is not None: echant.append((prev, c))
        prev = c; m.appliquer_action((0, 0))
    scores = {nom: np.mean([_rappel(_rendre(_decaler(mod.liste_objets(a), di, dj), t), b, t)
                            for a, b in echant]) for nom, (di, dj) in candidats.items()}
    meilleur = max(scores, key=scores.get)
    di, dj = candidats[meilleur]
    print(f"Décalage retenu : {meilleur} (scores {({k: round(x,2) for k,x in scores.items()})})")

    champ_prec = None; rappels = []
    for pas in range(300 if args.delai else 150):
        set_temps(step=pas)
        champ = m.percevoir()["vision"][-1]
        if champ_prec is not None:
            objets = mod.liste_objets(champ_prec)
            champ_pred = _rendre(_decaler(objets, di, dj), t)
            rappels.append(_rappel(champ_pred, np.asarray(champ), t))
            if args.log:
                log("agent", "pouls", t=pas, position=[int(m.agent_pos[0]), int(m.agent_pos[1])],
                    vitesse=list(v), action_choisie=[0, 0], inc_vision=round(mod.incertitude(), 4),
                    dyn_incertitude={}, n_predicteurs=len(objets), n_maitrises=0,
                    resolution=[1, t, t], champ_vu=_str(champ), champ_prevu=_str(champ_pred))
                if args.delai > 0 and logger is not None:
                    logger.f.flush(); import time; time.sleep(args.delai)
        champ_prec = champ
        m.appliquer_action((0, 0))

    r = sum(rappels) / len(rappels) if rappels else 0.0
    print(f"Prédiction TRIVIALE (objets décalés de la vitesse, AUCUN réseau) : rappel={r:.0%}")
    print("OK — latent structuré → prédiction triviale" if r > 0.5 else "à affiner")
    if args.log:
        print(f"Journal : {args.log} → python3 viewer.py --log {args.log}")


if __name__ == "__main__":
    main()
