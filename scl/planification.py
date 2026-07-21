"""Planification — le g() de A* ÉMERGE : un modèle de récompense appris (étape 18, §6).

`ModeleRecompense` apprend r̂(champ, action) = la récompense immédiate attendue d'une action
depuis un champ (gain sucre − douleur + progrès − temps, `pulsions.recompense`). C'est la
brique `g()` : le coût/gain d'un pas, prédit AVANT d'agir. Un planificateur glouton choisit
alors l'action de meilleure récompense prédite — d'abord un évitement de la douleur (le pas
suivant qui ne percute pas), avant que la recherche multi-pas (A*, étape 19) et le crédit
nocturne amont (étape 20) n'apprennent à VISER le sucre lointain.

On applique l'enseignement #10 (batch-1 en ligne ne converge pas) : mémoire de rejeu +
mini-lot. Aucune géométrie d'objet : l'entrée est le champ brut, la cible un scalaire mesuré.
"""
import random
from collections import deque

import numpy as np
import torch

from .config import CONFIG
from .logger import log
from .module_ae import DEVICE


class ModeleRecompense(torch.nn.Module):
    """r̂(champ, action) : champ aplati + action one-hot → récompense scalaire prédite."""

    def __init__(self, n_cellules=100, n_actions=5, h=128, capacite=4000):
        super().__init__()
        self.n_actions = n_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_cellules + n_actions, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, 1))
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=capacite)
        self.to(DEVICE)

    def _entree(self, champ, action_idx):
        c = torch.as_tensor(np.asarray(champ), dtype=torch.float32, device=DEVICE).reshape(-1)
        oh = torch.zeros(self.n_actions, device=DEVICE); oh[action_idx] = 1.0
        return torch.cat([c, oh])

    def predire(self, champ, action_idx):
        with torch.no_grad():
            return float(self.net(self._entree(champ, action_idx).unsqueeze(0)).squeeze())

    def observer(self, champ, action_idx, r, lot=32):
        """Ajoute (champ, action, r) à la mémoire et fait UN pas de mini-lot."""
        self.buffer.append((np.asarray(champ, dtype=np.float32).reshape(-1), action_idx, float(r)))
        n = min(len(self.buffer), lot)
        ech = random.sample(list(self.buffer), n)
        X = torch.stack([self._entree(c, a) for c, a, _ in ech])
        y = torch.tensor([[r] for _, _, r in ech], dtype=torch.float32, device=DEVICE)
        pred = self.net(X)
        perte = torch.nn.functional.mse_loss(pred, y)
        self.opt.zero_grad(); perte.backward(); self.opt.step()
        return float(perte.detach())


def choisir_glouton(modele_r, champ, n_actions, epsilon=0.1):
    """Action de meilleure récompense prédite (ε-glouton pour continuer d'explorer)."""
    if random.random() < epsilon:
        return random.randrange(n_actions)
    scores = [modele_r.predire(champ, a) for a in range(n_actions)]
    return int(np.argmax(scores))


def incertitude_module(module, fenetre=30):
    """Incertitude d'un module de transition = son erreur de prédiction récente moyenne.
    Élevée = conséquence mal prévue (à explorer) ; jamais essayé = max (attrait de l'inconnu)."""
    err = getattr(module, "erreurs", None)
    if not err:
        return 1.0
    return float(np.mean(err[-fenetre:]))


def choisir_curieux(tac, actions, epsilon=0.0):
    """CURIOSITÉ (étape 21) : choisir l'action dont on prévoit le MOINS bien la conséquence
    (incertitude max du modèle de transition action-conditionné). Quand plus rien ne
    s'apprend en exploitant (vol rectiligne), c'est ce qui sort l'agent de sa zone de
    confort : lister les actions, viser celle qu'on ne sait pas prévoir — puis la suivante,
    de proche en proche, jusqu'à tout maîtriser."""
    if random.random() < epsilon:
        return random.randrange(len(actions))
    inc = [incertitude_module(tac.modules[tuple(a)]) for a in actions]
    return int(np.argmax(inc))


class ModeleValeurQ(torch.nn.Module):
    """Q(champ, action) = g() + h() COMBINÉS (étape 19, §6) : la valeur d'agir, qui unit
    la récompense immédiate (g, le coût du pas) et la valeur du reste à faire (h, l'avenir).
    Apprise par TD (bootstrap) : Q(s,a) ← r + γ·max_a' Q(s',a'). C'est ce bootstrap qui fait
    REMONTER le crédit d'une récompense lointaine (sucre) vers les états/actions en amont —
    donc VISER le sucre, pas seulement éviter la douleur immédiate (étape 18). Tête façon
    DQN (champ → une valeur par action) : le max sur les actions est immédiat.

    Sans modèle pour la SÉLECTION (robuste au décalage de vitesse du modèle de transition) ;
    le modèle de transition `agir` sert au rejeu IMAGINÉ nocturne (étape 20)."""

    def __init__(self, n_cellules=100, n_actions=5, h=128, gamma=0.95, capacite=8000):
        super().__init__()
        self.n_actions = n_actions
        self.gamma = gamma
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_cellules, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, n_actions))
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.buffer = deque(maxlen=capacite)     # (s_flat, a, r, s2_flat)
        self.to(DEVICE)

    def _flat(self, champ):
        return torch.as_tensor(np.asarray(champ), dtype=torch.float32, device=DEVICE).reshape(-1)

    def q(self, champ):
        with torch.no_grad():
            return self.net(self._flat(champ).unsqueeze(0)).squeeze(0)

    def choisir(self, champ, epsilon=0.1):
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        return int(torch.argmax(self.q(champ)))

    def observer(self, s, a, r, s2, lot=64):
        """Ajoute la transition et fait un pas de TD sur un mini-lot (enseignement #10)."""
        self.buffer.append((np.asarray(s, np.float32).reshape(-1), int(a), float(r),
                            np.asarray(s2, np.float32).reshape(-1)))
        n = min(len(self.buffer), lot)
        ech = random.sample(list(self.buffer), n)
        S = torch.stack([torch.as_tensor(c, device=DEVICE) for c, _, _, _ in ech])
        A = torch.tensor([a for _, a, _, _ in ech], device=DEVICE).unsqueeze(1)
        R = torch.tensor([r for _, _, r, _ in ech], dtype=torch.float32, device=DEVICE)
        S2 = torch.stack([torch.as_tensor(c, device=DEVICE) for _, _, _, c in ech])
        q_sa = self.net(S).gather(1, A).squeeze(1)
        with torch.no_grad():
            cible = R + self.gamma * self.net(S2).max(1).values
        perte = torch.nn.functional.mse_loss(q_sa, cible)
        self.opt.zero_grad(); perte.backward(); self.opt.step()
        return float(perte.detach())
