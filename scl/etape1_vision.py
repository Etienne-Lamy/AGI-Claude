"""ÉTAPE 1 — validation ISOLÉE de la vision (leçon README v2 : valider le
module seul, sans monde vivant ni orchestrateur, avant toute intégration).

Fixe une vitesse NON NULLE (le champ défile), entraîne l'autoencodeur en ligne
à chaque pas sur GPU, et mesure DEUX choses :
  - MSE (peut être basse même si tout est faux — ne pas s'y fier seule) ;
  - RAPPEL objets = fraction des sucres/bâtons/corps réellement reconstruits
    (c'est LUI qui prouve qu'on ne s'est pas effondré à zéro).

Écrit aussi un journal `pouls` (champ vu vs reconstruit) pour le dashboard :
    python3 -m scl.etape1_vision --pas 4000 --log etape1.jsonl
    python3 viewer.py --log etape1.jsonl --port 8400   # VU vs PRÉVU en lettres
"""
import argparse

import numpy as np

from .config import CONFIG
from .logger import configurer, log, set_temps
from .module_ae import DEVICE, ModuleAutoencodeur
from .monde import Monde


def champ_vu_str(frame):
    return ",".join(f"{float(x):.2f}" for x in np.asarray(frame).reshape(-1))


def recon_str(recon):
    return ",".join(f"{float(x):.2f}" for x in recon.reshape(-1).tolist())


def _fidelite_moyenne(ae, frames):
    """Rappel/précision moyens sur un jeu FIXE de frames (métrique stable, non
    bruitée par un seul champ)."""
    r = [ae.fidelite(f) for f in frames]
    return (sum(d["rappel"] for d in r) / len(r),
            sum(d["precision"] for d in r) / len(r))


def valider(n_pas=4000, vitesse=(1, 1), graine=1, log_pouls=False, periode=200):
    monde = Monde(graine=graine)
    monde.vitesse = np.array(vitesse, dtype=np.int64)   # vitesse fixe non nulle
    t = CONFIG["taille_perception"]
    ae = ModuleAutoencodeur("vision")

    # jeu d'évaluation FIXE (frames non consécutives) pour une courbe stable
    frames_eval = []
    for _ in range(40):
        for _ in range(5):
            monde.appliquer_action((0, 0))
        frames_eval.append(np.asarray(monde.percevoir()["vision"][-1]).copy())

    jalons = []
    for pas in range(n_pas):
        set_temps(step=pas)
        frame = monde.percevoir()["vision"][-1]          # 10×10, champ courant
        ae.entrainer(frame)
        monde.appliquer_action((0, 0))                   # garde la vitesse fixe
        if pas % periode == 0 or pas == n_pas - 1:
            rappel, prec = _fidelite_moyenne(ae, frames_eval)
            jalons.append((pas, ae.incertitude(), rappel, prec))
            if log_pouls:
                recon = ae.reconstruire(frame)
                log("agent", "pouls", t=pas, position=[int(monde.agent_pos[0]),
                    int(monde.agent_pos[1])], vitesse=list(vitesse),
                    action_choisie=[0, 0], inc_vision=round(ae.incertitude(), 4),
                    dyn_incertitude={}, n_predicteurs=0, n_maitrises=0,
                    resolution=[1, t, t], champ_vu=champ_vu_str(frame),
                    champ_prevu=recon_str(recon), rappel=round(rappel, 3),
                    precision=round(prec, 3))
    return ae, jalons


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas", type=int, default=4000)
    p.add_argument("--vitesse", type=int, nargs=2, default=[1, 1])
    p.add_argument("--graine", type=int, default=1)
    p.add_argument("--log", type=str, default="")
    args = p.parse_args()
    if args.log:
        configurer(chemin=args.log)

    def f1(r, p):
        return 2 * r * p / (r + p) if (r + p) else 0.0

    print(f"Device : {DEVICE}")
    ae, jalons = valider(args.pas, tuple(args.vitesse), args.graine,
                         log_pouls=bool(args.log))
    print(f"{'pas':>6} {'err_cell':>9} {'rappel':>8} {'precision':>10} {'F1':>7}")
    for pas, err, rappel, prec in jalons:
        print(f"{pas:6d} {err:9.4f} {rappel:8.1%} {prec:10.1%} {f1(rappel, prec):7.1%}")
    _, _, r, p = jalons[-1]
    # moyenne des 5 derniers jalons (stable)
    q = jalons[-5:]
    rm = sum(x[2] for x in q) / len(q)
    pm = sum(x[3] for x in q) / len(q)
    f = f1(rm, pm)
    verdict = "OK — la vision reconstruit vraiment" if f > 0.7 \
        else "INSUFFISANT — reconstruction à améliorer"
    print(f"\nVERDICT étape 1 (moyenne 5 derniers) : rappel={rm:.0%} précision={pm:.0%} "
          f"F1={f:.0%} → {verdict}")


if __name__ == "__main__":
    main()
