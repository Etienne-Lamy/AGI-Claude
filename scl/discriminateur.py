"""Discriminateur SCL — D_φ, UN SEUL, partagé par tout le système et par le
rejeu nocturne (§0, §5) : jamais un discriminateur par module. Classification
générative-contrastive (NCE, Gutmann & Hyvärinen 2010) : validateur de
plausibilité pour le chaînage de générateurs (pipeline de création,
`graphe.py`) et pour la mémoire épisodique générative (`simulateur.py`,
`memoire_travail.py`).

Entrée générique : `evaluer_plausibilite`/`entrainer_contrastif` acceptent
des vecteurs de dimension quelconque (projection déterministe si nécessaire,
`utils.projeter`) — le discriminateur doit pouvoir juger des flux hétérogènes
issus de modules différents sans qu'on lui apprenne une dimension fixe par
consommateur.
"""
import math
import torch

from .config import CONFIG
from .logger import log, log_verbeux
from .utils import projeter


def attenuer_soft(poids, rang, lam=None):
    """w_j = exp(-λ r_j) — atténuation douce (shrinkage, James-Stein/LASSO),
    JAMAIS un masque à zéro (§5) : même à rang élevé, le poids reste
    strictement positif — un gradient qui pourrait redevenir utile n'est
    jamais définitivement tué."""
    lam = lam if lam is not None else CONFIG["lambda_attenuation"]
    return poids * math.exp(-lam * rang)


class Discriminateur:
    """Réseau unique, instancié une fois (par `inne.construire_graphe_inne`,
    Phase 11) et partagé par l'ensemble du système."""

    def __init__(self, dimension=None):
        self.dimension = dimension or CONFIG["dim_discriminateur"]
        d = self.dimension
        h = CONFIG["n_hidden_discriminateur"]
        self.W1 = torch.nn.Parameter(torch.randn(h, d) * (1.0 / max(1, d)) ** 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(h))
        self.W2 = torch.nn.Parameter(torch.randn(1, h) * (1.0 / max(1, h)) ** 0.5)
        self.b2 = torch.nn.Parameter(torch.zeros(1))
        self._g = [torch.zeros_like(p) for p in self.parametres()]
        log("discriminateur", "creation", dimension=self.dimension)

    def parametres(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def _logit(self, x):
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
        x = x.flatten().float()
        x = x if x.numel() == self.dimension else projeter(x, self.dimension)
        h = torch.relu(self.W1 @ x + self.b1)
        return (self.W2 @ h + self.b2).squeeze(0)

    def evaluer_plausibilite(self, x):
        """D_φ(x) ∈ (0,1) : probabilité que x appartienne à la réalité."""
        with torch.no_grad():
            p = torch.sigmoid(self._logit(x))
        log_verbeux("discriminateur", "evaluation", plausibilite=float(p))
        return float(p)

    def entrainer_contrastif(self, positif, negatifs, phase="jour"):
        """NCE : un positif réel contre N négatifs générés (§5). Les
        négatifs sont pondérés par atténuation douce selon leur rang dans la
        liste (jamais annulés)."""
        for p in self.parametres():
            p.grad = None
        perte = -torch.nn.functional.logsigmoid(self._logit(positif))
        for i, neg in enumerate(negatifs):
            poids = attenuer_soft(1.0, rang=i)
            perte = perte - poids * torch.nn.functional.logsigmoid(-self._logit(neg))
        perte = perte / (1 + len(negatifs))
        perte.backward()

        beta = CONFIG["beta_jour"] if phase == "jour" else CONFIG["beta_nuit"]
        for i, p in enumerate(self.parametres()):
            if p.grad is not None:
                self._g[i].mul_(beta).add_(p.grad, alpha=1 - beta)
        with torch.no_grad():
            for p, g in zip(self.parametres(), self._g):
                p -= CONFIG["lr_discriminateur"] * g

        e = float(perte.detach())
        log("discriminateur", "entrainement_contrastif", perte=e,
            n_negatifs=len(negatifs), phase=phase)
        return e
