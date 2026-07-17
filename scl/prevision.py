"""Modèle de prévision du corps (§1.3 "vitesse→image", §12 prédiction) —
apprentissage en ligne de la dynamique propre du corps : (vitesse, accélération)
→ vitesse au pas suivant. C'est ce que l'agent doit apprendre pour « maîtriser
son corps » : comment ses commandes se traduisent en mouvement, y compris la
saturation (v borné à ±v_max).

Une fois ce modèle fiable, la sélection d'action peut s'appuyer sur LUI plutôt
que sur la vérité-terrain de la physique (`boucle._scores_actions` mode
`appris`) : l'agent navigue alors avec son propre modèle appris du monde. La
perception étant égocentrée (le corps est au centre, le monde défile de −v'),
la position relative d'un sucre au pas suivant est `rel − v'` — le seul inconnu
appris est v', la réponse du corps à la commande.

Auto-supervisé et strictement local (§2 : aucun gradient global) : cible = la
vitesse réellement observée après l'action, disponible gratuitement au pas
suivant. Accumulateur de gradient à moment, deux cadences jour/nuit (§1.3),
comme les autres modules.
"""
import torch

from .config import CONFIG
from .logger import log, log_verbeux


class ModelePrevisionCorps:
    """MLP (vx, vy, ax, ay) → (v'x, v'y). Petit, entraîné en ligne."""

    def __init__(self, v_max=None):
        self.v_max = v_max if v_max is not None else CONFIG["v_max"]
        h = CONFIG["n_hidden_prevision"]
        self.W1 = torch.nn.Parameter(torch.randn(h, 4) * (1.0 / 4) ** 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(h))
        self.W2 = torch.nn.Parameter(torch.randn(2, h) * (1.0 / max(1, h)) ** 0.5)
        self.b2 = torch.nn.Parameter(torch.zeros(2))
        self._g = [torch.zeros_like(p) for p in self.parametres()]
        self.n_maj = 0
        self.erreur_recente = []   # fenêtre pour π/fiabilité
        log("prevision", "creation", v_max=self.v_max)

    def parametres(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def _forward(self, x):
        h = torch.relu(self.W1 @ x + self.b1)
        return self.W2 @ h + self.b2

    def predire(self, v, accel):
        """v' prédit (couple d'entiers arrondis, borné à ±v_max) pour la
        commande `accel` depuis la vitesse `v`."""
        x = torch.tensor([float(v[0]), float(v[1]),
                          float(accel[0]), float(accel[1])])
        with torch.no_grad():
            sortie = self._forward(x)
        vp = torch.clamp(sortie.round(), -self.v_max, self.v_max)
        return (int(vp[0]), int(vp[1]))

    def apprendre(self, v, accel, v_observe, phase="jour"):
        """Un pas de descente sur (v, accel) → v_observe (vitesse réellement
        constatée après l'action). Retourne l'erreur MSE."""
        x = torch.tensor([float(v[0]), float(v[1]),
                          float(accel[0]), float(accel[1])])
        cible = torch.tensor([float(v_observe[0]), float(v_observe[1])])
        for p in self.parametres():
            p.grad = None
        sortie = self._forward(x)
        erreur = torch.mean((sortie - cible) ** 2)
        erreur.backward()
        beta = CONFIG["beta_jour"] if phase == "jour" else CONFIG["beta_nuit"]
        for i, p in enumerate(self.parametres()):
            if p.grad is not None:
                self._g[i].mul_(beta).add_(p.grad, alpha=1 - beta)
        with torch.no_grad():
            for p, g in zip(self.parametres(), self._g):
                p -= CONFIG["lr_prevision"] * g
        e = float(erreur.detach())
        self.n_maj += 1
        self.erreur_recente.append(e)
        if len(self.erreur_recente) > CONFIG["fenetre_fiabilite_prevision"]:
            self.erreur_recente.pop(0)
        log_verbeux("prevision", "apprentissage", erreur=e, phase=phase, n_maj=self.n_maj)
        return e

    def fiabilite(self):
        """π ∈ [0,1] : confiance dans le modèle appris, dérivée de l'erreur
        récente (§1.4). Tant qu'il n'a jamais appris : 0 (dégénérescence,
        l'instinct garde la main). Sert au transfert progressif instinct →
        appris (§15.1 fusion par confiance)."""
        if self.n_maj < CONFIG["n_maj_min_prevision"] or not self.erreur_recente:
            return 0.0
        err = sum(self.erreur_recente) / len(self.erreur_recente)
        return 1.0 / (1.0 + err / CONFIG["echelle_fiabilite_prevision"])
