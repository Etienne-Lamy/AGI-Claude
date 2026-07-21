"""Modèle de transition ACTION-CONDITIONNÉ (étape 16, §3 de la conception).

L'action devient prévisible comme le reste : un module champ→champ DÉDIÉ à chaque action
prédit `champ(t+1)` sachant `champ(t)` et l'action émise. Ce n'est pas de la triche que de
conditionner sur sa propre action : l'agent connaît la commande qu'il envoie (copie
d'efférence / proprioception). On réutilise `ModuleAutoencodeur` en mode transition
(prouvé étapes 2a/10) — aucun nouveau bas-niveau.

`agir(champ, a)` rend le futur IMAGINABLE : c'est la brique sur laquelle l'arbre de
planification A* (étapes 18-19) déroulera les conséquences des actions sans toucher au
monde. Ici on mesure l'effet d'UN pas d'action (l'accumulation multi-pas — vitesse qui
grandit — se compose ensuite avec la dynamique du corps `dynamique.py`).
"""
from .config import CONFIG
from .logger import log
from .module_ae import ModuleAutoencodeur


class TransitionActionChamp:
    """Un prédicteur de transition champ→champ par action (copie d'efférence)."""

    def __init__(self, actions):
        self.actions = [tuple(a) for a in actions]
        self.modules = {a: ModuleAutoencodeur(f"agir_{a[0]}_{a[1]}") for a in self.actions}
        self.n_maj = {a: 0 for a in self.actions}
        log("action", "creation_transition_action", actions=[list(a) for a in self.actions])

    def observer(self, champ_prec, action, champ):
        """Entraîne le module de l'action émise sur la transition réelle observée."""
        a = tuple(action)
        e = self.modules[a].entrainer_transition(champ_prec, champ)
        self.n_maj[a] += 1
        return e

    def predire(self, champ_prec, action):
        """Champ prédit sous `action` — le futur imaginé (aucun accès au monde)."""
        return self.modules[tuple(action)].predire(champ_prec)

    def rappel(self, champ_prec, action, champ):
        """Rappel objets ∈ [0,1] de la prédiction de `action` sur la transition."""
        return self.modules[tuple(action)].fidelite_transition(champ_prec, champ)["rappel"]

    def matrice_rappel(self, transitions):
        """Matrice croisée : rappel du module de l'action `a` sur les transitions
        réellement produites par l'action `b`. `transitions` : liste
        (champ_prec, action_reelle, champ). Retourne {a: {b: rappel_moyen}}."""
        import numpy as np
        par_b = {b: [] for b in self.actions}
        for cp, b, c in transitions:
            par_b[tuple(b)].append((cp, c))
        mat = {}
        for a in self.actions:
            mat[a] = {}
            for b in self.actions:
                rs = [self.modules[a].fidelite_transition(cp, c)["rappel"] for cp, c in par_b[b]]
                mat[a][b] = float(np.mean(rs)) if rs else 0.0
        return mat
