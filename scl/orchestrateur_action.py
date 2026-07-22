"""Orchestrateur d'ACTION à attention, entraîné par A* (étape 26, §4 du plan).

Concrètement, la structure que réclamait l'auteur, instanciée sur la navigation :

  - JETONS : un par objet perçu. `token = [ one-hot(catégorie) | x, y | confiance ]` — la
    catégorie et la confiance sont les SIGNAUX FAIBLES (ici : type d'objet = sucre/bâton, et
    fiabilité de la perception). Même rôle que « activation / condensateur / fiabilité » des
    jetons‑modules d'`attention.py`.
  - ENCODEUR : Set Transformer (self-attention multi-tête Q/K/V/W_O, SANS position — invariance
    par permutation). La matrice d'attention `A` dit quels objets comptent l'un pour l'autre.
  - TÊTE : pooling des jetons + vitesse → logits d'action.
  - ENTRAÎNEMENT PAR A* : A* (`chercher_sucre`) fournit la PREMIÈRE action optimale de chaque
    état → imitation (entropie croisée). Le gradient traverse tête → attention → projection des
    jetons. L'orchestrateur AMORTIT la recherche : à l'inférence il émet l'action SANS dérouler
    l'arbre. (REINFORCE au-delà de l'imitation = étape suivante.)

Aucun gradient n'entre dans les modules de perception : l'orchestrateur n'apprend qu'à LIRE
l'état‑objet et ses signaux pour agir (séparation §10.8).
"""
import numpy as np
import torch

from .module_ae import DEVICE
from .monde import ACCELERATIONS_PERMISES


class OrchestrateurAction(torch.nn.Module):
    def __init__(self, k_categories, d=64, n_tetes=4, n_actions=5, avec_type=True, avec_conf=True):
        super().__init__()
        self.K = k_categories
        self.avec_type, self.avec_conf = avec_type, avec_conf
        self.f_in = k_categories + 2 + 1          # one-hot(K) + (x,y) + confiance
        self.n_tetes = n_tetes; self.d = d; self.d_t = d // n_tetes
        s = lambda: torch.nn.Linear(d, d, bias=False)
        self.proj = torch.nn.Linear(self.f_in, d)
        self.Wq, self.Wk, self.Wv, self.Wo = s(), s(), s(), s()
        self.ff = torch.nn.Sequential(torch.nn.Linear(d, 4 * d), torch.nn.ReLU(), torch.nn.Linear(4 * d, d))
        self.tete = torch.nn.Sequential(torch.nn.Linear(d + 2, d), torch.nn.ReLU(), torch.nn.Linear(d, n_actions))
        self.opt = torch.optim.Adam(self.parameters(), lr=2e-3)
        self.to(DEVICE)

    def _attention(self, X):                      # X : (N, d) → (N, d), équivariant
        N = X.shape[0]
        Q = self.Wq(X).view(N, self.n_tetes, self.d_t).transpose(0, 1)
        K = self.Wk(X).view(N, self.n_tetes, self.d_t).transpose(0, 1)
        V = self.Wv(X).view(N, self.n_tetes, self.d_t).transpose(0, 1)
        A = torch.softmax(Q @ K.transpose(-2, -1) / self.d_t ** 0.5, dim=-1)
        H = (A @ V).transpose(0, 1).reshape(N, self.d)
        H = X + self.Wo(H)
        return H + self.ff(H)

    def logits(self, tokens, v):
        """tokens : (N, f_in) tenseur ; v : (2,). Retourne les logits d'action (n_actions)."""
        vv = torch.tensor([float(v[0]) / 2, float(v[1]) / 2], device=DEVICE)
        if tokens.shape[0] == 0:
            pooled = torch.zeros(self.d, device=DEVICE)
        else:
            pooled = self._attention(self.proj(tokens)).mean(0)
        return self.tete(torch.cat([pooled, vv]))

    def choisir(self, tokens, v):
        with torch.no_grad():
            return int(torch.argmax(self.logits(tokens, v)))

    def imiter(self, lot):
        """lot : liste de (tokens, v, action_idx) — imite la première action d'A*."""
        self.opt.zero_grad()
        perte = torch.zeros((), device=DEVICE)
        for tokens, v, a in lot:
            lg = self.logits(tokens, v).unsqueeze(0)
            perte = perte + torch.nn.functional.cross_entropy(
                lg, torch.tensor([a], device=DEVICE))
        perte = perte / max(1, len(lot))
        perte.backward(); self.opt.step()
        return float(perte.detach())


def tokens_objets(po, objets, orch):
    """Construit les jetons (N, f_in). Ablations : `avec_type` (one-hot catégorie), `avec_conf`."""
    if not objets:
        return torch.zeros((0, orch.f_in), device=DEVICE)
    t, cen = po.t, po.centre
    lignes = []
    for k, (i, j) in objets:
        oh = np.zeros(po.clf.K, dtype=np.float32)
        if orch.avec_type:
            oh[k] = 1.0                            # SIGNAL FAIBLE : type d'objet (sucre/bâton…)
        conf = 1.0 if orch.avec_conf else 0.0      # placeholder de fiabilité de perception
        lignes.append(np.concatenate([oh, [(i - cen) / t, (j - cen) / t, conf]]).astype(np.float32))
    return torch.tensor(np.stack(lignes), device=DEVICE)
