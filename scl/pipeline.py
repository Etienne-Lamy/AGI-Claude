"""Construction du pipeline hiérarchique complet, partagé par les harnais.

    N0 champ → [compresseur] → N1 latent → [modules-vitesse] → N2 régime actif
                                          → [module transition] → N3 règle action→régime

Rend l'ensemble prêt à l'emploi (compresseur gelé, vocabulaire de régimes verrouillé,
règle N3 entraînée) pour que chaque étape se concentre sur ce qu'elle démontre.
"""
import random

import numpy as np

from .composition import DetecteurVitesse
from .hierarchie import ModuleTransitionRegime
from .logger import set_temps
from .module_ae import ModuleAutoencodeur
from .monde import Monde

VITESSES = [(0, 0), (1, 0), (2, 0), (-1, 0), (-2, 0)]
ACTIONS = [(0, 0), (1, 0), (-1, 0)]


def construire(pas_comp=2000, pas_regime=900, pas_action=2500, graine=1, verbose=True):
    """Retourne (comp, det, n3, idx, noms, monde, assoc)."""
    comp = ModuleAutoencodeur("compresseur")
    m = Monde(graine=graine); m.vitesse = np.array([1, 0], dtype=np.int64)
    if verbose:
        print("compresseur…", flush=True)
    for s in range(pas_comp):
        set_temps(step=s); comp.entrainer(m.percevoir()["vision"][-1]); m.appliquer_action((0, 0))

    # --- vocabulaire de régimes (N2) : un module-vitesse par vitesse, puis verrouillé
    det = DetecteurVitesse(comp)
    assoc = {}
    if verbose:
        print("vocabulaire de régimes…", flush=True)
    for v in VITESSES:
        det.delai.pousser(None)
        m.vitesse = np.array(v, dtype=np.int64)
        for s in range(pas_regime):
            set_temps(step=s); det.etape(m.percevoir()["vision"][-1]); m.appliquer_action((0, 0))
        det.delai.pousser(None)
        scores = {}
        for _ in range(60):
            actif, _, _ = det.identifier(m.percevoir()["vision"][-1]); m.appliquer_action((0, 0))
            if actif:
                scores[actif] = scores.get(actif, 0) + 1
        assoc[v] = max(scores, key=scores.get) if scores else None
    for mv in det.vitesses.values():
        mv.verrouille = True

    ids = sorted(det.vitesses)
    idx = {vid: i for i, vid in enumerate(ids)}
    inv = {vid: v for v, vid in assoc.items() if vid}
    noms = [str(inv.get(vid, "?")) for vid in ids]

    # --- règle N3 : (régime, action) → régime suivant
    n3 = ModuleTransitionRegime("n3_transition", len(ids), len(ACTIONS))
    m.vitesse = np.array([0, 0], dtype=np.int64)
    det.delai.pousser(None)
    prec = None
    if verbose:
        print("règle N3 (action → régime)…", flush=True)
    for s in range(pas_action):
        set_temps(step=s)
        a = random.choice(ACTIONS)
        actif, _, _ = det.identifier(m.percevoir()["vision"][-1])
        m.appliquer_action(a)
        if prec is not None and actif is not None:
            n3.entrainer(idx[prec[0]], ACTIONS.index(prec[1]), idx[actif])
        prec = (actif, a) if actif is not None else None
    return comp, det, n3, idx, noms, m, assoc
