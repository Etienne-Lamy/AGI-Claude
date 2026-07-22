"""ÉTAPE 24 — action = ACCÉLÉRATION + compositionnalité (2,0)=(1,0)∘(1,0) (§2 du plan).

On prédit sur l'état-objet sous des SÉQUENCES d'accélérations (vraies branches, pas un
`(1,0)×10` figé), et on démontre que la vitesse (2,0) se simule par DOUBLE usage du décalage
de (1,0) — aucun module (2,0) dédié.

    python3 -m scl.etape24_action_objet
"""
import argparse

import numpy as np

from .config import CONFIG
from .dynamique_objet import DynamiqueObjet
from .logger import set_temps
from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES, Monde
from .perception_objet import ChampObjet

VITESSES_ENTRAINEMENT = [(1, 0), (2, 0), (0, 1), (1, 1), (0, 0), (-1, 0), (0, -1)]


def _rappel_prev(pred, cible, shift, t, cen, seuil=0.2):
    """Rappel sur la région PRÉVISIBLE : cellules cibles dont la source (i+sx,j+sy) était
    en vue (les objets entrant par le bord ne sont pas prévisibles)."""
    pred, cible = np.asarray(pred), np.asarray(cible)
    sx, sy = int(shift[0]), int(shift[1])
    obj, ok, n = None, 0, 0
    for i in range(t):
        for j in range(t):
            si, sj = i + sx, j + sy
            if 0 <= si < t and 0 <= sj < t and cible[i, j] > seuil:
                n += 1
                ok += int(abs(pred[i, j] - cible[i, j]) < seuil)
    return ok / n if n else 1.0


def _entrainer(po, pas):
    m = Monde(graine=1); rng = np.random.default_rng(0)
    for s in range(pas):
        set_temps(step=s)
        po.entrainer(np.asarray(m.percevoir()["vision"][-1]).copy())
        if s % 15 == 0:
            m.vitesse = np.array(VITESSES_ENTRAINEMENT[int(rng.integers(len(VITESSES_ENTRAINEMENT)))], dtype=np.int64)
        m.appliquer_action((0, 0))
    po.calibrer()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=4000)
    p.add_argument("--eval", type=int, default=60)
    args = p.parse_args()
    import torch; torch.manual_seed(0)
    print(f"Device : {DEVICE} — action=accélération sur l'état-objet")

    po = ChampObjet(); _entrainer(po, args.pas)
    dyn = DynamiqueObjet(po)
    t, cen = po.t, po.centre

    # --- TEST A : compositionnalité (2,0) = (1,0)∘(1,0) ---
    print("\nA. Compositionnalité : simuler v=(2,0) par DOUBLE usage du décalage de (1,0)")
    eg, ec, rreel = [], [], []
    m = Monde(graine=77); m.vitesse = np.array([2, 0], dtype=np.int64)
    for _ in range(args.eval):
        cp = np.asarray(m.percevoir()["vision"][-1]).copy()
        objs = po.objets(cp)
        direct = po.regenerer(po.decaler(objs, (2, 0)))            # module (2,0) direct
        compose = po.regenerer(dyn.translater_compose(objs, (2, 0)))  # (1,0) appliqué 2×
        m.appliquer_action((0, 0)); reel = np.asarray(m.percevoir()["vision"][-1]).copy()
        # égalité direct vs composé (doit être identique)
        eg.append(float(np.mean(np.abs(np.asarray(direct) - np.asarray(compose)) < 1e-6)))
        ec.append(_rappel_prev(compose, reel, (2, 0), t, cen))     # composé vs réalité
        rreel.append(_rappel_prev(direct, reel, (2, 0), t, cen))
    print(f"   décalage (2,0) direct == (1,0)∘(1,0) : {np.mean(eg):.0%} des cellules identiques")
    print(f"   (1,0)∘(1,0) vs réalité (prévisible) : {np.mean(ec):.0%}  ; (2,0) direct : {np.mean(rreel):.0%}")
    compo_ok = np.mean(eg) > 0.999 and np.mean(ec) > 0.95

    # --- TEST B : prédiction multi-pas sous SÉQUENCES d'actions (vraies branches) ---
    print("\nB. Prédiction multi-pas sous séquences d'accélérations variées :")
    rng = np.random.default_rng(1)
    H = 6
    rec_par_pas = np.zeros(H); nseq = 20
    for _ in range(nseq):
        m = Monde(graine=int(rng.integers(1, 10000))); m.vitesse = np.zeros(2, np.int64)
        E0 = po.objets(np.asarray(m.percevoir()["vision"][-1]).copy())
        actions = [ACCELERATIONS_PERMISES[int(rng.integers(len(ACCELERATIONS_PERMISES)))] for _ in range(H)]
        traj = dyn.derouler(E0, (0, 0), actions)
        shift = np.zeros(2, int)
        for h, a in enumerate(actions):
            m.appliquer_action(a); reel = np.asarray(m.percevoir()["vision"][-1]).copy()
            shift = shift + np.array(traj[h][1])            # décalage cumulé = somme des vitesses
            pred = po.regenerer(traj[h][0])
            rec_par_pas[h] += _rappel_prev(pred, reel, tuple(shift), t, cen)
    rec_par_pas /= nseq
    print("   rappel prévisible par pas : " + "  ".join(f"T+{h+1}:{rec_par_pas[h]:.0%}" for h in range(H)))

    # branches : deux séquences différentes → deux futurs différents. On mesure la divergence
    # SUR LES CELLULES-OBJETS (le vide identique ne compte pas), moyennée sur plusieurs départs
    # ayant au moins 2 objets (sinon rien à distinguer).
    rng2 = np.random.default_rng(7); diffs = []
    while len(diffs) < 15:
        m = Monde(graine=int(rng2.integers(1, 10000))); m.vitesse = np.zeros(2, np.int64)
        E0 = po.objets(np.asarray(m.percevoir()["vision"][-1]).copy())
        if len(E0) < 2:
            continue
        f1 = np.asarray(po.regenerer(dyn.derouler(E0, (0, 0), [(1, 0)] * 4)[-1][0]))
        f2 = np.asarray(po.regenerer(dyn.derouler(E0, (0, 0), [(0, 1)] * 4)[-1][0]))
        obj = (f1 > CONFIG["seuil_objet_vision"]) | (f2 > CONFIG["seuil_objet_vision"])
        n = int(obj.sum())
        if n:
            diffs.append(float(((np.abs(f1 - f2) > 1e-6) & obj).sum()) / n)
    diff = float(np.mean(diffs))
    print(f"   branches (1,0)×4 vs (0,1)×4 : {diff:.0%} des cellules-OBJETS diffèrent → futurs distincts")

    ok = compo_ok and rec_par_pas[:3].mean() > 0.9 and diff > 0.5
    print(f"\n{'OK' if ok else 'à affiner'} — l'action est une accélération ; (2,0)=(1,0)∘(1,0) "
          f"vérifié ; prédiction multi-pas exacte (prévisible) ; branches distinctes"
          if ok else f"\nà affiner")


if __name__ == "__main__":
    main()
