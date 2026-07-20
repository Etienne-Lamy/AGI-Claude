"""ÉTAPE 12 — boucle JOUR→NUIT sur un imprévu (§31.7-31.8).

JOUR : l'agent connaît un régime (vocabulaire formé, verrouillé). Un VENT se lève
(régime inconnu) → la familiarité s'effondre → l'ENREGISTREUR capture l'épisode
surprenant, et le détecteur fait NAÎTRE un module en temps réel (budget serré).

NUIT : on REJOUE l'épisode et on tente de le comprendre — un module dédié apprend
ses transitions jusqu'à les prédire. Critère mesurable de « COMPRIS » : le rejeu est
désormais prédit (rappel > seuil). L'épisode quitte la liste des choses à comprendre.

    python3 -m scl.etape12_jour_nuit
"""
import argparse

import numpy as np

from .logger import set_temps
from .memoire_episodique import Enregistreur, MemoireEpisodique
from .module_ae import DEVICE
from .monde import Monde
from .nuit import travailler_la_nuit
from .regime import DetecteurRegimeChamp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pas_regime", type=int, default=2500)
    p.add_argument("--pas_jour", type=int, default=400)
    p.add_argument("--vent", type=int, nargs=2, default=[0, 2])
    args = p.parse_args()
    print(f"Device : {DEVICE}")

    # --- JOUR (avant vent) : former puis verrouiller le vocabulaire connu
    det = DetecteurRegimeChamp()
    m = Monde(graine=1); m.vitesse = np.array([1, 0], dtype=np.int64)
    print("Formation du régime connu (v=(1,0))…")
    for s in range(args.pas_regime):
        set_temps(step=s); det.etape(m.percevoir()["vision"][-1]); m.appliquer_action((0, 0))
    det.verrouiller_tout()
    n_avant = len(det.regimes)
    print(f"   {n_avant} module(s) connu(s) : {list(det.regimes)}")

    # --- JOUR : le VENT se lève (régime inconnu)
    memoire = MemoireEpisodique()
    enr = Enregistreur()
    m.vent = np.array(args.vent, dtype=np.int64)
    det.champ_prec = None
    familiarites = []
    print(f"\nJOUR — le vent {tuple(args.vent)} se lève ({args.pas_jour} pas)…")
    for s in range(args.pas_jour):
        set_temps(step=s)
        champ = m.percevoir()["vision"][-1]
        actif, _, fam = det.identifier(champ)
        familiarites.append(fam)
        ep = enr.observer(champ, (0, 0), actif, fam)
        if ep is not None:
            memoire.enregistrer(ep)
        det.etape(champ)                         # temps réel : peut faire naître un module
        m.appliquer_action((0, 0))
    # scelle un éventuel épisode encore ouvert (le vent ne s'arrête pas)
    ep = enr._en_cours
    if ep is not None and len(ep.champs) >= 5:
        memoire.enregistrer(ep)

    fam_moy = float(np.mean([f for f in familiarites if f]))
    print(f"   familiarité moyenne sous vent : {fam_moy:.0%}")
    print(f"   modules nés en temps réel     : {len(det.regimes) - n_avant}")
    print(f"   épisodes surprenants capturés : {len(memoire.episodes)}")
    if not memoire.episodes:
        print("\n(aucun épisode capturé — la surprise n'a pas franchi le seuil)")
        return

    # --- NUIT : comprendre les épisodes
    print(f"\nNUIT — rejeu et compréhension de {len(memoire.a_comprendre())} épisode(s)…")
    appris = travailler_la_nuit(memoire)
    for e in memoire.episodes:
        print(f"   épisode (durée {len(e.champs)}, familiarité min {e.familiarite_min:.0%}) : "
              f"{'COMPRIS' if e.compris else 'non compris'}")
    print(f"\nModules appris cette nuit : {len(appris)}")
    ok = len(appris) >= 1 and all(e.compris for e in memoire.episodes)
    print("\nOK — l'imprévu du jour a été CAPTURÉ puis COMPRIS la nuit"
          if ok else "\nà affiner — épisode non compris au rejeu")


if __name__ == "__main__":
    main()
