"""Mémoire épisodique (§29.5, §31.7-31.8) — ne retenir que le NON-RÉGÉNÉRABLE.

Ce que les modules savent prédire est déjà « stocké » dans les modules. Un épisode
mémorisé ne contient donc que sa GRAINE de régénération :
    1. le champ initial (état de départ) ;
    2. la séquence d'ACTIONS ;
    3. l'identité des MODULES actifs (le régime) ;
    4. le RÉSIDU (la seule part que la chaîne n'a pas su prédire).

On ne mémorise QUE les épisodes surprenants (familiarité effondrée). Critère de
« COMPRIS » (§31.8), mesurable : l'épisode est régénérable depuis sa graine ET
désormais prédit (rappel du meilleur module au-dessus d'un seuil sur le rejeu).
"""
from dataclasses import dataclass, field

import numpy as np

from .config import CONFIG
from .logger import log


@dataclass
class Episode:
    champ_initial: np.ndarray
    actions: list                       # séquence d'accélérations subies/choisies
    modules_actifs: list                # identités des régimes actifs (par pas)
    champs: list                        # champs réels observés (résidu = ce qui surprend)
    familiarite_min: float              # à quel point c'était inattendu
    compris: bool = field(default=False)


class MemoireEpisodique:
    def __init__(self, capacite=None):
        self.capacite = capacite or CONFIG["capacite_memoire_episodique"]
        self.episodes = []

    def enregistrer(self, episode):
        """Ajoute un épisode surprenant ; purge les plus anciens COMPRIS d'abord
        (on garde ce qui reste à comprendre)."""
        self.episodes.append(episode)
        if len(self.episodes) > self.capacite:
            for i, e in enumerate(self.episodes):
                if e.compris:
                    del self.episodes[i]
                    break
            else:
                del self.episodes[0]
        log("memoire", "episode_enregistre", n_episodes=len(self.episodes),
            familiarite_min=round(episode.familiarite_min, 3), duree=len(episode.champs))

    def a_comprendre(self):
        return [e for e in self.episodes if not e.compris]

    def marquer_compris(self, episode, rappel_rejeu):
        episode.compris = True
        log("memoire", "episode_compris", rappel_rejeu=round(rappel_rejeu, 3),
            restant=len(self.a_comprendre()))


class Enregistreur:
    """Capture l'épisode en cours quand une SURPRISE se déclare (§31.7). Tant que la
    familiarité reste basse, on accumule ; à la sortie de surprise, on scelle."""

    def __init__(self, seuil_surprise=None):
        # HYSTÉRÉSIS (§29.1, obligatoire) : on ENTRE en surprise sous `seuil_bas`,
        # on n'en SORT qu'au-dessus de `seuil_haut`. Sans ça, la familiarité bruitée
        # franchit le seuil dans les deux sens et FRAGMENTE un imprévu continu en
        # dizaines de mini-épisodes (mesuré : 1 vent → ~20 épisodes).
        self.seuil_bas = seuil_surprise if seuil_surprise is not None else CONFIG["seuil_familiarite_surprise"]
        self.seuil_haut = self.seuil_bas + CONFIG["hysteresis_surprise"]
        self._en_cours = None

    def observer(self, champ, action, module_actif, familiarite):
        """À appeler à chaque pas. Retourne un Episode SCELLÉ si la surprise vient de
        se terminer, sinon None."""
        if self._en_cours is None:
            surpris = familiarite < self.seuil_bas          # entrée en surprise
        else:
            surpris = familiarite < self.seuil_haut         # on reste tant qu'on n'est pas revenu au familier
        if surpris:
            if self._en_cours is None:
                self._en_cours = Episode(
                    champ_initial=np.asarray(champ).copy(), actions=[],
                    modules_actifs=[], champs=[], familiarite_min=familiarite)
            e = self._en_cours
            e.actions.append(tuple(int(a) for a in action))
            e.modules_actifs.append(module_actif)
            e.champs.append(np.asarray(champ).copy())
            e.familiarite_min = min(e.familiarite_min, familiarite)
            return None
        # fin de surprise : on scelle si l'épisode a une durée minimale
        if self._en_cours is not None and len(self._en_cours.champs) >= CONFIG["duree_min_episode"]:
            scelle, self._en_cours = self._en_cours, None
            return scelle
        self._en_cours = None
        return None
