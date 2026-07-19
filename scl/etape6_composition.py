"""ÉTAPE 6 — ÉMERGENCE d'une composition de modules qui DÉTECTE la vitesse.

Chaîne : compresseur(champ)→z ; délai→z(T-1) ; module-vitesse→ẑ(T) ; comparé au
latent réel z(T) ET, via le générateur du compresseur, au champ réel.

On fait défiler plusieurs vitesses. À chaque vitesse NOUVELLE, les modules
existants échouent (surprise) → un module-vitesse NAÎT et se spécialise. Ensuite
on mesure la matrice de détection : chaque module a un résidu FAIBLE à SA vitesse
(diagonale) et élevé aux autres → la composition détecte la vitesse, aux deux
niveaux (latent et champ).

    python3 -m scl.etape6_composition --pas_comp 3000 --pas_regime 1500
"""
import argparse

import numpy as np
import torch

from .composition import DetecteurVitesse, _residu_champ
from .config import CONFIG
from .logger import set_temps
from .module_ae import DEVICE, ModuleAutoencodeur
from .monde import Monde

VITESSES = [(1, 1), (2, 0), (0, 2)]


def _flux(vitesse, graine, n):
    m = Monde(graine=graine); m.vitesse = np.array(vitesse, dtype=np.int64)
    for _ in range(n):
        yield m.percevoir()["vision"][-1]
        m.appliquer_action((0, 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas_comp", type=int, default=2000)
    p.add_argument("--pas_regime", type=int, default=800)
    p.add_argument("--pas_eval", type=int, default=40)
    args = p.parse_args()

    # 1) compresseur (module 1) entraîné puis GELÉ
    comp = ModuleAutoencodeur("compresseur")
    print(f"Device : {DEVICE} — entraînement du compresseur…")
    for s, champ in enumerate(_flux((1, 1), 1, args.pas_comp)):
        set_temps(step=s); comp.entrainer(champ)

    # 2) détecteur : on fait défiler les régimes ; les modules émergent
    det = DetecteurVitesse(comp)
    print("Défilement des régimes (émergence des modules-vitesse) :")
    for v in VITESSES:
        det.delai.pousser(None)
        actifs_avant = len(det.vitesses)
        res, res_fin = [], []
        for s, champ in enumerate(_flux(v, 7, args.pas_regime)):
            _, r = det.etape(champ)
            if r:
                m = min(x[0] for x in r.values())
                res.append(m)
                if s > args.pas_regime * 0.8:
                    res_fin.append(m)
        n_nes = len(det.vitesses) - actifs_avant
        print(f"   vitesse {v}: {len(det.vitesses)} modules (+{n_nes} né(s)) | "
              f"résidu min moyen {np.mean(res):.4f} → fin de régime {np.mean(res_fin):.4f}")

    # 3) matrice de détection, calculée UNE SEULE FOIS : pour chaque vitesse, on
    # déroule un flux frais et on note le résidu de CHAQUE module (latent + champ).
    ids = list(det.vitesses)
    matL = {vid: {} for vid in ids}          # résidu latent   [module][vitesse]
    matC = {vid: {} for vid in ids}          # résidu champ    [module][vitesse]
    for v in VITESSES:
        acc = {vid: ([], []) for vid in ids}
        det.delai.pousser(None)
        for champ in _flux(v, 123, args.pas_eval):
            z = det.comp.encoder(champ).detach()
            if det.delai.sortie is not None:
                for vid in ids:
                    zpred = det.vitesses[vid].predire(det.delai.sortie)
                    acc[vid][0].append(float(torch.mean((zpred - z) ** 2)))
                    acc[vid][1].append(_residu_champ(det.comp, zpred, champ))
            det.delai.pousser(z)
        for vid in ids:
            matL[vid][v] = float(np.mean(acc[vid][0]))
            matC[vid][v] = float(np.mean(acc[vid][1]))

    print("\nMatrice de détection — résidu LATENT (bas = ce module prédit ce régime) :")
    print("            " + "  ".join(f"{str(v):>9}" for v in VITESSES))
    couvertes = set()
    for vid in ids:
        ligne = "  ".join(f"{matL[vid][v]:9.4f}" for v in VITESSES)
        gagne = min(VITESSES, key=lambda v: matL[vid][v])
        couvertes.add(gagne)
        print(f"{vid:>11} {ligne}   → régime détecté {gagne}")

    print("\nMatrice — résidu CHAMP (via le générateur ; 1−rappel objets) :")
    print("            " + "  ".join(f"{str(v):>9}" for v in VITESSES))
    for vid in ids:
        print(f"{vid:>11} " + "  ".join(f"{matC[vid][v]:9.3f}" for v in VITESSES))

    print(f"\nModules nés : {len(ids)} | vitesses distinctes couvertes : "
          f"{len(couvertes)}/{len(VITESSES)}")
    print("OK — des modules-vitesse ont ÉMERGÉ et la composition détecte la vitesse"
          if len(couvertes) >= 2 else "à affiner")


if __name__ == "__main__":
    main()
