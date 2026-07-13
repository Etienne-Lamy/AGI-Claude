"""Simulateur SCL — S_new, mémoire épisodique GÉNÉRATIVE (§10.2, §8.3).

On ne stocke pas l'épisode fondateur, on stocke une MACHINE capable de le
refabriquer : c'est ce qui rend le rejeu nocturne possible à partir d'un
seul exemple réel (cohérence CLS, §11). Instancié 1:1 avec un module créé
par le pipeline de création jumelée (`graphe.creer_module_candidat`).

Contrainte non négociable (§10.2) : tête hétéroscédastique OBLIGATOIRE — un
simulateur sans incertitude conditionnelle ne rejoue pas ; sans elle, la
reconstruction ponctuelle d'un exemple unique serait une mémorisation
triviale qui passe D_φ sans généraliser."""
import torch

from .config import CONFIG
from .logger import log, log_verbeux
from .utils import ajuster_dim


class Simulateur:
    def __init__(self, id, dim_contexte_echec, dim_latent_stocke=None):
        self.id = id
        self.dim_contexte_echec = dim_contexte_echec
        self.dim_latent_stocke = dim_latent_stocke or CONFIG["dim_emb"]
        h = CONFIG["n_hidden_simulateur"]
        self.W1 = torch.nn.Parameter(
            torch.randn(h, self.dim_latent_stocke) * (1.0 / max(1, self.dim_latent_stocke)) ** 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(h))
        self.W_mu = torch.nn.Parameter(torch.randn(dim_contexte_echec, h) * (1.0 / max(1, h)) ** 0.5)
        self.b_mu = torch.nn.Parameter(torch.zeros(dim_contexte_echec))
        self.W_sigma = torch.nn.Parameter(torch.randn(dim_contexte_echec, h) * (1.0 / max(1, h)) ** 0.5)
        self.b_sigma = torch.nn.Parameter(torch.zeros(dim_contexte_echec))
        self._g = [torch.zeros_like(p) for p in self.parametres()]
        self.z_fondateur = None   # z_stocké de l'épisode fondateur (§10.2)
        log(self.id, "creation_simulateur", dim_contexte_echec=dim_contexte_echec,
            dim_latent_stocke=self.dim_latent_stocke)

    def parametres(self):
        return [self.W1, self.b1, self.W_mu, self.b_mu, self.W_sigma, self.b_sigma]

    def _forward(self, z):
        """(μ, σ) — tête hétéroscédastique, σ toujours strictement positif.
        `log_sigma` est borné (clamp) : sans cela, la perte de
        vraisemblance gaussienne peut effondrer σ vers 0 avant que μ n'ait
        convergé (le terme (x-μ)²/σ² explose alors), un mode d'échec connu
        de l'entraînement hétéroscédastique — détecté ici précisément parce
        que ce fichier est validé en isolation avant toute intégration."""
        z = ajuster_dim(z, self.dim_latent_stocke)
        h = torch.relu(self.W1 @ z + self.b1)
        mu = self.W_mu @ h + self.b_mu
        log_sigma = torch.clamp(self.W_sigma @ h + self.b_sigma, min=-4.0, max=4.0)
        sigma = torch.exp(log_sigma)
        return mu, sigma

    def initialiser_depuis_episode(self, z_fondateur, x_echec, n_pas=None):
        """Ancre le simulateur sur l'épisode fondateur RÉEL (un seul exemple
        au moment de la création jumelée) — quelques pas d'apprentissage
        immédiats, le rejeu nocturne affine ensuite."""
        self.z_fondateur = z_fondateur.detach().clone()
        n_pas = n_pas or CONFIG["n_iterations_alignement"]
        for _ in range(n_pas):
            self.entrainer(self.z_fondateur, x_echec)
        log(self.id, "initialisation_episode_fondateur", n_pas=n_pas)

    def entrainer(self, z, x_cible, phase="jour"):
        """Log-vraisemblance négative gaussienne (tête hétéroscédastique
        obligatoire, §10.2)."""
        for p in self.parametres():
            p.grad = None
        mu, sigma = self._forward(z)
        x_cible = ajuster_dim(x_cible, mu.numel()).detach()
        nll = torch.mean(0.5 * ((x_cible - mu) / sigma) ** 2 + torch.log(sigma))
        nll.backward()
        # la perte NLL hétéroscédastique peut exploser localement quand σ
        # devient petit (1/σ² amplifie le gradient) — sans ce clip, un seul
        # pas peut faire diverger μ et σ simultanément (observé en isolation,
        # exactement ce que cette porte de validation est censée détecter).
        torch.nn.utils.clip_grad_norm_(self.parametres(), CONFIG["clip_grad_simulateur"])
        beta = CONFIG["beta_jour"] if phase == "jour" else CONFIG["beta_nuit"]
        grads = [p.grad for p in self.parametres()]
        for i, g in enumerate(grads):
            if g is not None:
                self._g[i].mul_(beta).add_(g, alpha=1 - beta)
        with torch.no_grad():
            for p, g in zip(self.parametres(), self._g):
                p -= CONFIG["lr_simulateur"] * g
        e = float(nll.detach())
        log_verbeux(self.id, "entrainement_simulateur", nll=e, phase=phase)
        return e

    def refabriquer(self, z_stocke=None):
        """Régénère un contexte stocké (brut de capteurs ou latent d'un
        module amont certifié) — (μ, Σ), potentiellement purement latent.
        Sans z fourni : rejoue l'épisode fondateur (§10.2, §10.7)."""
        z = z_stocke if z_stocke is not None else self.z_fondateur
        if z is None:
            raise ValueError("aucun z fourni et aucun épisode fondateur enregistré")
        with torch.no_grad():
            mu, sigma = self._forward(z)
        return mu, sigma

    def generer_contrefactuel(self, chemin_reel, discriminateur, n=None):
        """Rollout de chemins non empruntés à partir d'un succès réel γ+ —
        augmentation dream/nightmare (§8.3). Perturbe le chemin réel dans
        l'espace latent stocké, échantillonne via la tête hétéroscédastique,
        et fait juger la plausibilité par D_φ (jamais accepté aveuglément)."""
        n = n or CONFIG["n_contrefactuels"]
        z_base = ajuster_dim(chemin_reel, self.dim_latent_stocke).detach()
        variantes = []
        for _ in range(n):
            bruit = torch.randn(self.dim_latent_stocke) * CONFIG["echelle_bruit_contrefactuel"]
            z_variante = z_base + bruit
            with torch.no_grad():
                mu, sigma = self._forward(z_variante)
                echantillon = mu + sigma * torch.randn_like(mu)
                plausibilite = discriminateur.evaluer_plausibilite(echantillon)
            variantes.append({"chemin": echantillon, "plausibilite": plausibilite,
                              "provenance": "imagine"})
        log(self.id, "generer_contrefactuel", n=n,
            plausibilite_moyenne=sum(v["plausibilite"] for v in variantes) / n)
        return variantes

    def est_hors_distribution(self, contexte, discriminateur, seuil=None):
        """Signale que le verdict de D_φ doit être étiqueté HYPOTHÈSE, pas
        PILIER, pour ce contexte (§10.2) : D_φ a été entraîné sur l'ancienne
        distribution — face au radicalement neuf, sa propre plausibilité est
        elle-même hors distribution."""
        seuil = seuil if seuil is not None else CONFIG["seuil_hors_distribution"]
        p = discriminateur.evaluer_plausibilite(contexte)
        hors_distribution = p < seuil
        log_verbeux(self.id, "est_hors_distribution", plausibilite=p,
                    hors_distribution=hors_distribution)
        return hors_distribution
