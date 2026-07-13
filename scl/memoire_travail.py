"""Mémoire de travail SCL (§11, §12) : ligne à retard à décalage relatif.

Non négociable (§0) : AUCUNE indexation par horaire absolu — toute mémoire
temporelle est indexée par offset relatif à l'instant courant. `TamponRelatif`
ne stocke ni ne consulte jamais de compteur de temps absolu ; seul le
décalage δ∈{-K,...,K} porte l'information temporelle, et il est réinitialisé
implicitement à chaque `decaler`."""
import torch

from .config import CONFIG
from .logger import log, log_verbeux
from .utils import ajuster_dim


class TamponRelatif:
    """Tampon circulaire de taille 2K+1, indexé par offset relatif
    δ∈{-K,...,K} — JAMAIS par horaire absolu (§0, §11.1)."""

    def __init__(self, K):
        self.K = K
        self.taille = 2 * K + 1
        self._contenu = [None] * self.taille   # index interne = δ + K

    def _idx(self, delta):
        if not (-self.K <= delta <= self.K):
            raise IndexError(f"offset {delta} hors de [-{self.K}, {self.K}]")
        return delta + self.K

    def lire(self, delta):
        return self._contenu[self._idx(delta)]

    def ecrire(self, delta, valeur):
        self._contenu[self._idx(delta)] = valeur

    def decaler(self, nouvelle_observation):
        """Décale tous les offsets d'un tick : le contenu à δ devient celui
        à δ-1. Le contenu sortant à δ=-K est perdu. Le contenu qui vient
        d'atteindre δ=0 (la prédiction arrivée à échéance) est renvoyé AVANT
        d'être écrasé par `nouvelle_observation` — c'est le point d'ancrage
        de la maturation (§12) : chaque prédiction sert deux fois, une fois
        à l'émission, une fois ici."""
        self._contenu = self._contenu[1:] + [None]
        arrive_a_maturite = self._contenu[self.K]   # δ=0, avant écrasement
        self.ecrire(0, nouvelle_observation)
        return arrive_a_maturite


def hierarchie_deux_vitesses(K_rapide=None, K_lent=None):
    """Hiérarchie à deux vitesses (§11.1) : tête rapide (bornée par W,
    quelques emplacements) + palier lent plus grand — analogie
    registre/mémoire principale d'un calculateur. Retourne deux
    `TamponRelatif` indépendants (rapide, lent)."""
    K_rapide = K_rapide if K_rapide is not None else CONFIG["W"]
    K_lent = K_lent if K_lent is not None else CONFIG["K"]
    return TamponRelatif(K_rapide), TamponRelatif(K_lent)


def fenetre_glissante_continue(tampon, reel_courant):
    """Pas de replanification complète : décale la fenêtre, le masquage se
    lève tick par tick au rythme où le réel remplace le prévu (§12). AUCUNE
    nouveauté par rapport à `TamponRelatif.decaler` — le document source le
    dit explicitement ("clarification seule") : cette fonction nomme le
    concept pour l'appelant, elle n'ajoute pas de mécanisme."""
    return tampon.decaler(reel_courant)


class FamilleHorizon:
    """Famille d'horizon de prédiction (§12) : émet une prédiction à
    l'offset relatif +h, à cadence propre à la famille, compare au réel à
    maturation (δ=0). POC : une seule famille court terme (h fixe) — les
    familles moyen/long terme (§4.4, `statistiques.cadence_variable`) sont
    additives plus tard, sans restructuration."""

    def __init__(self, nom, h, K=None, cadence=1):
        self.nom = nom
        self.h = h
        self.cadence = cadence
        self.tampon = TamponRelatif(K if K is not None else max(h, 1))
        self._pas_depuis_emission = 0

    def emettre(self, module, entree, t=0):
        """Émet une prédiction à l'offset relatif +h — cadence propre à la
        famille (ne prédit pas à chaque pas si `cadence` > 1). `t` n'est
        qu'une métadonnée de journalisation transportée avec la prédiction,
        jamais utilisée pour indexer le tampon."""
        self._pas_depuis_emission += 1
        if self._pas_depuis_emission < self.cadence:
            return None
        self._pas_depuis_emission = 0
        with torch.no_grad():
            latent = module.forward_reconnaissance(entree)
            prediction = module.forward_generation(latent)[: module.n_outputs_gen]
        self.tampon.ecrire(self.h, {"prediction": prediction.detach(), "t_emission": t})
        log_verbeux(self.nom, "emission_prediction", h=self.h, t=t)
        return prediction

    def maturer(self, reel, t=0):
        """Décale le tampon d'un tick ; si une prédiction atteignait
        l'échéance (δ=0), compare au réel et renvoie le résidu (consommé par
        `statistiques.sprt_surprise`)."""
        arrivee = self.tampon.decaler(reel)
        if arrivee is None:
            return None
        prediction = arrivee["prediction"]
        cible = ajuster_dim(reel, prediction.numel())
        residu = float(torch.mean((cible - prediction) ** 2))
        log(self.nom, "maturation_prediction", residu=residu,
            t_emission=arrivee["t_emission"], t=t)
        return residu


class PalierSommeil:
    """Palier mémoire supplémentaire (§10.7) : contextes non résolus,
    maintenus hors mémoire de calcul rapide (opérateur idle/NOP, réemployé
    — pas de mécanisme dédié) jusqu'à récupération nocturne, sous
    précondition de reconstruction jugée suffisante par D_φ."""

    def __init__(self):
        self._stockes = []   # liste de dicts : z (latent, potentiellement pur), t, resolu

    def stocker(self, z, t=0):
        self._stockes.append({"z": z, "t": t, "resolu": False})
        log_verbeux("palier_sommeil", "stockage", t=t, n_total=len(self._stockes))

    def recuperer(self, simulateur, discriminateur, seuil_plausibilite=None):
        """Récupère UN contexte stocké non résolu pour rejeu nocturne, sous
        précondition de reconstruction jugée suffisante par D_φ (§10.7) :
        refabrique via le simulateur, ne renvoie que si la reconstruction
        est plausible. Renvoie le contexte refabriqué (μ) ou None."""
        seuil = (seuil_plausibilite if seuil_plausibilite is not None
                else CONFIG["seuil_hors_distribution"])
        for entree in self._stockes:
            if entree["resolu"]:
                continue
            mu, _sigma = simulateur.refabriquer(entree["z"])
            plausibilite = discriminateur.evaluer_plausibilite(mu)
            if plausibilite >= seuil:
                entree["resolu"] = True
                log("palier_sommeil", "recuperation", t=entree["t"],
                    plausibilite=plausibilite)
                return mu
            log_verbeux("palier_sommeil", "recuperation_refusee",
                        t=entree["t"], plausibilite=plausibilite)
        return None
