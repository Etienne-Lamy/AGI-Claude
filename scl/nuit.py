"""Cycle NOCTURNE de l'orchestrateur (§31.8) — comprendre l'imprévu au calme.

Le jour, un épisode surprenant est capturé (memoire_episodique.Enregistreur) mais
pas maîtrisé (budget temps réel serré). La nuit, sans contrainte de temps, on le
REJOUE et on RÉ-ESSAIE de le prédire — en entraînant longuement un module dédié sur
ses transitions. Critère de « COMPRIS » (mesurable) : le rejeu est désormais prédit
(rappel au-dessus du seuil). Ce module rejoint la mémoire à long terme (dormant),
réactivable si la situation revient (§29.1, §29.5).

Générique : rien ici ne sait qu'il s'agit de « vent » ou de vitesse — on apprend à
reproduire/prédire une séquence de champs jusqu'à ce qu'elle cesse de surprendre.
"""
import numpy as np

from .config import CONFIG
from .logger import log
from .module_ae import ModuleAutoencodeur


def rejouer_et_comprendre(episode, n_passes=None):
    """Entraîne un module DÉDIÉ à prédire les transitions de l'épisode (plusieurs
    passes = temps calme de la nuit), puis mesure s'il le prédit. Retourne
    (module, rappel_rejeu, compris)."""
    n_passes = n_passes or CONFIG["passes_nuit"]
    champs = episode.champs
    if len(champs) < 2:
        return None, 0.0, False
    m = ModuleAutoencodeur("nuit_module")
    for _ in range(n_passes):
        for i in range(len(champs) - 1):
            m.entrainer_transition(champs[i], champs[i + 1])
    rappels = [m.fidelite_transition(champs[i], champs[i + 1])["rappel"]
               for i in range(len(champs) - 1)]
    r = float(np.mean(rappels)) if rappels else 0.0
    compris = r > CONFIG["seuil_rappel_compris"]
    log("nuit", "rejeu_episode", rappel_rejeu=round(r, 3), compris=compris,
        n_transitions=len(champs) - 1)
    return m, r, compris


def travailler_la_nuit(memoire, modules_appris=None):
    """Pour chaque épisode NON compris : le rejouer, tenter de le comprendre, et —
    si réussi — le marquer compris et garder le module (dormant, réactivable).
    Retourne la liste des modules appris cette nuit."""
    modules_appris = modules_appris if modules_appris is not None else []
    for episode in list(memoire.a_comprendre()):
        module, r, compris = rejouer_et_comprendre(episode)
        if compris:
            memoire.marquer_compris(episode, r)
            modules_appris.append(module)
    log("nuit", "cycle_termine", modules_appris=len(modules_appris),
        episodes_restants=len(memoire.a_comprendre()))
    return modules_appris
