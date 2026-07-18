"""ÉTAPE 2 (enchaînement) — prédiction du champ visuel SUIVANT par la chaîne
module 1 → module 2 → module 1, à vitesse fixe, comparée au réel.

    champ_{t-1} --enc(module 1)--> z_{t-1} --module 2--> ẑ_t --gen(module 1)--> champ_t PRÉDIT
    vs champ_t RÉEL (capteur).

Entraîne d'abord le module 1 (vision, déjà validé étape 1), puis GÈLE son
encodeur/décodeur et entraîne le module 2 dans son espace abstrait (un module
GPU à la fois → évite le watchdog Kepler). Écrit un journal `pouls` pour le
dashboard : `champ_vu` = réel(t), `champ_prevu` = prédit(t) par la chaîne.

    python3 -m scl.etape2b_chaine --vitesse 1 1 --log etape2.jsonl
    python3 viewer.py --log etape2.jsonl --port 8400   # RÉEL vs PRÉDIT-suivant
"""
import argparse

import numpy as np

from .config import CONFIG
from .logger import configurer, log, set_temps
from .module_ae import DEVICE, ModuleAutoencodeur, PredicteurAbstrait
from .monde import Monde


def _champ_str(f):
    return ",".join(f"{float(x):.2f}" for x in np.asarray(f).reshape(-1))


def _recall(pred_field, vrai_field, seuil=0.1):
    """Rappel objets de la prédiction chaînée vs le vrai champ."""
    import torch
    p = pred_field.reshape(-1)
    v = torch.as_tensor(vrai_field, dtype=torch.float32).reshape(-1)
    obj = v > seuil
    n = int(obj.sum())
    if not n:
        return 1.0
    juste = ((p - v).abs() < 0.2) & obj
    return int(juste.sum()) / n


def entrainer(vitesse=(1, 1), graine=1, pas_vision=1500, pas_pred=1500):
    m = Monde(graine=graine)
    m.vitesse = np.array(vitesse, dtype=np.int64)

    # --- module 1 : vision (étape 1) ---
    vision = ModuleAutoencodeur("vision")
    for s in range(pas_vision):
        set_temps(step=s)
        vision.entrainer(m.percevoir()["vision"][-1])
        m.appliquer_action((0, 0))

    # --- module 2 : prédicteur abstrait, module 1 GELÉ ---
    pred = PredicteurAbstrait("pred_abstrait")
    z_prec = None
    for s in range(pas_pred):
        set_temps(step=pas_vision + s)
        champ = m.percevoir()["vision"][-1]
        z = vision.encoder(champ)                 # champ abstrait (no grad côté module 1)
        if z_prec is not None:
            pred.entrainer(z_prec.detach().cpu(), z.detach().cpu())
        z_prec = z
        m.appliquer_action((0, 0))
    return vision, pred, m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vitesse", type=int, nargs=2, default=[1, 1])
    p.add_argument("--graine", type=int, default=1)
    p.add_argument("--log", type=str, default="")
    p.add_argument("--pas_demo", type=int, default=120)
    args = p.parse_args()
    if args.log:
        configurer(chemin=args.log)

    v = tuple(args.vitesse)
    print(f"Device : {DEVICE} — entraînement (vitesse {v})…")
    vision, pred, m = entrainer(vitesse=v, graine=args.graine)

    # démo : dérouler la chaîne et comparer prédit(t) vs réel(t)
    t = CONFIG["taille_perception"]
    champ_prec = None
    recalls = []
    for pas in range(args.pas_demo):
        set_temps(step=pas)
        champ = m.percevoir()["vision"][-1]
        if champ_prec is not None:
            z_prec = vision.encoder(champ_prec)
            z_pred = pred.predire(z_prec)                 # module 2
            champ_pred = vision.generer(z_pred).cpu()     # module 1 (générateur)
            recalls.append(_recall(champ_pred, champ))
            if args.log:
                log("agent", "pouls", t=pas,
                    position=[int(m.agent_pos[0]), int(m.agent_pos[1])],
                    vitesse=list(v), action_choisie=[0, 0],
                    inc_vision=round(vision.incertitude(), 4), dyn_incertitude={},
                    n_predicteurs=1, n_maitrises=0, resolution=[1, t, t],
                    champ_vu=_champ_str(champ),                 # RÉEL(t)
                    champ_prevu=_champ_str(champ_pred))         # PRÉDIT(t) via chaîne
        champ_prec = champ
        m.appliquer_action((0, 0))

    r = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"Rappel de la prédiction CHAÎNÉE (modules 1→2→1) à vitesse {v} : {r:.0%}")
    print("OK — la chaîne prédit le champ suivant" if r > 0.7
          else "à revoir — prédiction chaînée insuffisante")
    if args.log:
        print(f"Journal écrit : {args.log} → `python3 viewer.py --log {args.log}`")


if __name__ == "__main__":
    main()
