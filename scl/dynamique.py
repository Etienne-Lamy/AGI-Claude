"""Dynamique du corps — prédicteurs (vitesse → vitesse suivante) créés À LA
DEMANDE, un par accélération, quand agir révèle une surprise, puis entraînés
jusqu'à maîtrise. Support de l'émergence motrice par curiosité (§4, §15.2).

Prior inné trivial : « rien ne change » (v_suivant ≈ v). Pour l'accélération
nulle, ce prior est exact → maîtrisé d'emblée, aucun module. Pour une
accélération qui bouge réellement la vitesse, le résidu vs ce prior est une
SURPRISE ; accumulée et confirmée par SPRT (§4.5), elle fait naître un module
prédicteur DÉDIÉ à cette accélération. La curiosité pousse alors l'agent vers
les accélérations dont il ne prédit pas encore l'effet (incertitude haute),
jusqu'à les maîtriser toutes — « de proche en proche ».

Aucune coordonnée d'objet, aucune géométrie de tâche : l'agent n'apprend que la
conséquence de ses propres commandes sur son propre corps.
"""
from collections import deque

import torch

from . import curiosite
from .config import CONFIG
from .logger import log, log_verbeux
from .module import Module
from .statistiques import residu_normalise, sprt_creation


class Dynamique:
    """Ensemble de prédicteurs de dynamique, indexés par accélération."""

    def __init__(self):
        self.predicteurs = {}          # accel (tuple) → Module (v → v_suivant)
        self.surprises = {}            # accel → liste (contexte v, résidu) pour SPRT
        self.residu_baseline = {}      # accel → deque des ||v_suivant − v|| récents

    # ------------------------------------------------------ incertitude/curiosité
    def incertitude_action(self, v, accel):
        """Incertitude attendue de la CONSÉQUENCE de `accel` depuis la vitesse
        `v` — signal de curiosité (§15.2). Prédicteur dédié → son erreur
        récente ; sinon résidu baseline observé (l'accél. nulle y est ~0 =
        maîtrisée) ; jamais observée → attrait d'exploration (inconnu)."""
        if accel in self.predicteurs:
            return curiosite.incertitude(self.predicteurs[accel])
        base = self.residu_baseline.get(accel)
        if base:
            return sum(base) / len(base)
        return float(CONFIG["attrait_action_inexploree"])

    # ------------------------------------------------------------- apprentissage
    def observer(self, v_avant, accel, v_apres, t=0, phase="jour"):
        """Enregistre la transition réelle (v_avant, accel) → v_apres :
        entraîne le prédicteur dédié s'il existe, sinon accumule la surprise vs
        le prior « rien ne change » et crée un prédicteur si le SPRT confirme.
        Retourne l'erreur/résidu du pas."""
        v_avant = torch.tensor([float(v_avant[0]), float(v_avant[1])])
        v_apres = torch.tensor([float(v_apres[0]), float(v_apres[1])])

        if accel in self.predicteurs:
            e = self.predicteurs[accel].entrainer_predictif(
                v_avant, v_apres, contexte_vec=v_avant, t=t, phase=phase)
            log_verbeux("dynamique", "entrainement_predicteur", accel=list(accel), erreur=e)
            return e

        # pas encore de prédicteur : deux quantités DISTINCTES sur la même
        # transition, vs le prior trivial « v_suivant = v ».
        #  (1) erreur quadratique moyenne (échelle des prédicteurs → curiosité)
        mse_base = float(torch.mean((v_apres - v_avant) ** 2))
        self.residu_baseline.setdefault(accel, deque(maxlen=CONFIG["fenetre_incertitude"]))
        self.residu_baseline[accel].append(mse_base)
        #  (2) résidu NORMALISÉ (échelle χ²_d, ce que le SPRT attend) : un vrai
        #      changement de vitesse devient une surprise > d ; l'accél. nulle
        #      reste ~0. Sans cette normalisation, ‖Δv‖ brut passe pour non
        #      surprenant et aucun prédicteur ne naît (bug corrigé).
        surprise = residu_normalise(v_apres, v_avant, CONFIG["sigma_prior_dynamique"])
        self.surprises.setdefault(accel, []).append((v_avant, surprise))

        # surprise confirmée (contextes distincts, §4.5) → création dédiée
        decision, _ = sprt_creation(self.surprises[accel], d=2)
        if decision == "H1":
            self._creer_predicteur(accel, v_avant, v_apres, t, phase)
            self.surprises[accel] = []
        elif decision == "H0":
            self.surprises[accel] = []   # pas de dynamique à apprendre ici (ex. accél. nulle)
        return mse_base

    def _creer_predicteur(self, accel, v_avant, v_apres, t, phase):
        mid = f"dyn_{accel[0]}_{accel[1]}"
        m = Module(mid, n_inputs_reco=2, n_latent=CONFIG["n_latent_dynamique"],
                   n_outputs_gen=2)
        m.entrainer_predictif(v_avant, v_apres, contexte_vec=v_avant, t=t, phase=phase)
        self.predicteurs[accel] = m
        log("dynamique", "creation_predicteur", accel=list(accel), module=mid,
            n_predicteurs=len(self.predicteurs))

    # ------------------------------------------------------------------ rapport
    def etat_maitrise(self):
        """Pour l'instrumentation : {accel: (incertitude, maîtrisé?)} sur les
        accélérations déjà rencontrées."""
        rapport = {}
        for accel, m in self.predicteurs.items():
            rapport[accel] = (round(curiosite.incertitude(m), 4), curiosite.maitrise(m))
        return rapport
