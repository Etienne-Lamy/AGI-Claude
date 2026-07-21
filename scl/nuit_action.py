"""Rejeu nocturne AMONT pour la valeur d'action (étape 20, §7 de la conception).

Le jour, la récompense (contact d'un sucre) est trop RARE pour que le TD en ligne apprenne
à viser (étape 19 : gain dans le bruit). La nuit, on REJOUE les épisodes en PRIORISANT les
rares épisodes récompensés, avec des RETOURS n-PAS : le crédit du sucre remonte alors sur
les n états qui le précèdent — puis, nuit après nuit, de plus en plus en amont. C'est
« on mange d'abord un sucre par hasard, la nuit apprend à s'en approcher, de 1 pas puis 2… ».

Aucune géométrie d'objet : on ne rejoue que des transitions (champ, action, récompense)
déjà vécues ; le retour n-pas et la priorisation concentrent l'apprentissage, ils ne
fabriquent pas de signal.
"""
import random

import numpy as np
import torch

from .logger import log
from .module_ae import DEVICE


class RejeuNocturne:
    """Mémoire d'épisodes (listes de (champ, action, récompense)) rejoués la nuit."""

    def __init__(self, n_pas=6, priorite_sucre=8.0):
        self.n_pas = n_pas                 # profondeur du retour n-pas
        self.priorite_sucre = priorite_sucre   # sur-échantillonnage des épisodes récompensés
        self.episodes = []                 # liste de (episode, poids)

    def enregistrer(self, episode):
        """episode : liste de (champ, action, récompense). Pondéré ↑ s'il contient une
        récompense positive (un sucre) — c'est ce qu'on veut rejouer en priorité."""
        recompense = any(r > 0 for _, _, r in episode)
        poids = self.priorite_sucre if recompense else 1.0
        self.episodes.append((episode, poids))
        return recompense

    def _cibles_nstep(self, q, episode):
        """Retours n-pas pour chaque pas de l'épisode : G_t = Σ γ^i r_{t+i} + γ^m maxQ(s_{t+m})."""
        L = len(episode)
        champs = [torch.as_tensor(np.asarray(s, np.float32), device=DEVICE).reshape(-1)
                  for s, _, _ in episode]
        with torch.no_grad():
            qmax = q.net(torch.stack(champs)).max(1).values   # maxQ(s) pour tout l'épisode
        cibles, entrees, actions = [], [], []
        for t in range(L):
            m = min(self.n_pas, L - t)          # nb de récompenses sommées (≥1 : inclut r_t)
            G = 0.0
            for i in range(m):
                G += (q.gamma ** i) * episode[t + i][2]
            if t + m < L:                        # bootstrap sur l'état APRÈS les m récompenses
                G += (q.gamma ** m) * float(qmax[t + m])
            # sinon : fin d'épisode, pas de bootstrap (la récompense terminale EST incluse)
            entrees.append(champs[t]); actions.append(episode[t][1]); cibles.append(G)
        return torch.stack(entrees), torch.tensor(actions, device=DEVICE).unsqueeze(1), \
            torch.tensor(cibles, dtype=torch.float32, device=DEVICE)

    def nuit(self, q, passes=1500, lot=64):
        """Rejeu offline : échantillonne des épisodes (priorité sucre), calcule les retours
        n-pas, entraîne Q. Retourne la perte moyenne."""
        if not self.episodes:
            return 0.0
        eps, poids = zip(*self.episodes)
        pertes = []
        X, A, Y = [], [], []
        for _ in range(passes):
            e = random.choices(eps, weights=poids, k=1)[0]
            xe, ae, ye = self._cibles_nstep(q, e)
            X.append(xe); A.append(ae); Y.append(ye)
            if sum(x.shape[0] for x in X) >= lot:
                Xb, Ab, Yb = torch.cat(X), torch.cat(A), torch.cat(Y)
                q_sa = q.net(Xb).gather(1, Ab).squeeze(1)
                perte = torch.nn.functional.mse_loss(q_sa, Yb)
                q.opt.zero_grad(); perte.backward(); q.opt.step()
                pertes.append(float(perte.detach())); X, A, Y = [], [], []
        moy = float(np.mean(pertes)) if pertes else 0.0
        log("nuit", "rejeu_action", n_episodes=len(self.episodes), passes=passes, perte=round(moy, 4))
        return moy
