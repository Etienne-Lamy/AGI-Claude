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
