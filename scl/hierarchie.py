"""Niveau N3 — l'ACCÉLÉRATION comme règle apprise sur le signal de régime (§29.2).

Une fois qu'il existe un module-vitesse par régime (N2 = « quel module explique la
transition »), ce signal est DISCRET et de très faible cardinalité. On lui applique
le même outillage qu'à n'importe quel signal : le prédire. Mais ici la prédiction
n'a de sens que **conditionnée par l'action** :

    (régime à T-1 = v1)  +  action A  →  (régime à T = v2)

C'est exactement l'accélération — apprise comme un **modèle de transition sur
l'espace des modules**, jamais câblée ni calculée par un capteur. Le module est
générique (MLP sur one-hot ⊕ one-hot) : rien ne suppose que le signal soit une
vitesse ; il apprendrait de la même façon tout effet d'action sur un régime.

Comparaison honnête : le **prior trivial** ici est « le régime ne change pas »
(souvent vrai — l'accélération nulle domine). Le gain se lit donc surtout sur les
pas où le régime CHANGE réellement.
"""
import torch

from .config import CONFIG
from .logger import log, log_verbeux
from .module_ae import DEVICE


class ModuleTransitionRegime:
    """N3 : (régime, action) → régime suivant. Sortie = distribution sur régimes."""

    def __init__(self, id, n_regimes, n_actions, dim_cachee=64):
        self.id = id
        self.n_regimes, self.n_actions = n_regimes, n_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_regimes + n_actions, dim_cachee), torch.nn.ReLU(),
            torch.nn.Linear(dim_cachee, dim_cachee), torch.nn.ReLU(),
            torch.nn.Linear(dim_cachee, n_regimes)).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=3e-3)
        self.n_maj = 0
        log(self.id, "creation_module_transition_regime",
            n_regimes=n_regimes, n_actions=n_actions, device=str(DEVICE))

    def _entree(self, reg, act):
        x = torch.zeros(self.n_regimes + self.n_actions, device=DEVICE)
        x[reg] = 1.0
        x[self.n_regimes + act] = 1.0
        return x

    def predire(self, reg, act):
        with torch.no_grad():
            return int(self.net(self._entree(reg, act)).argmax())

    def entrainer(self, reg, act, reg_suivant):
        logits = self.net(self._entree(reg, act))
        perte = torch.nn.functional.cross_entropy(
            logits.unsqueeze(0), torch.tensor([reg_suivant], device=DEVICE))
        self.opt.zero_grad(); perte.backward(); self.opt.step()
        self.n_maj += 1
        log_verbeux(self.id, "entrainement_transition_regime", perte=float(perte.detach()))
        return float(perte.detach())

    def table(self, noms_regimes=None, noms_actions=None):
        """Règle APPRISE, lisible : pour chaque (régime, action) → régime prédit.
        C'est la forme interprétable de « quelle action fait passer de v1 à v2 »."""
        t = {}
        for r in range(self.n_regimes):
            for a in range(self.n_actions):
                cle = (noms_regimes[r] if noms_regimes else r,
                       noms_actions[a] if noms_actions else a)
                p = self.predire(r, a)
                t[cle] = noms_regimes[p] if noms_regimes else p
        return t


def gain_vs_trivial(predictions, verites, precedents):
    """Gain de prédictibilité (§28.1) contre le prior trivial « le régime ne change
    pas ». Retourne (exactitude, exactitude_triviale, gain, n_changements)."""
    n = len(verites)
    if not n:
        return 0.0, 0.0, 0.0, 0
    just = sum(1 for p, v in zip(predictions, verites) if p == v) / n
    triv = sum(1 for pr, v in zip(precedents, verites) if pr == v) / n
    err, err_t = 1.0 - just, 1.0 - triv
    gain = 1.0 - err / err_t if err_t > 1e-9 else 0.0
    n_chg = sum(1 for pr, v in zip(precedents, verites) if pr != v)
    return just, triv, gain, n_chg
