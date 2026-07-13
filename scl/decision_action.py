"""Décision d'action SCL (§15) — remplacement DIRECT de l'ancien
`orchestrateur.py`, qui violait le §0/§15.3 en mélangeant les besoins de
façon continue (`souhaitabilite_torch`) au lieu d'un seul besoin dominant
par argmax+hystérésis. Deux mécanismes bien distincts, à ne jamais confondre :

- `fusion_ponderee` : CONTINU, pondère perception réelle et génération
  prédite — c'est le seul endroit où un mélange continu est légitime (§15.1).
- `priorisation_besoin_dominant` : DISCRET, un seul besoin gouverne l'action
  à la fois (§15.3, argmax+hystérésis déjà implémenté dans
  `memoires.TableBesoins.besoin_dominant`) — jamais un mélange pondéré."""
import torch

from .config import CONFIG
from .logger import log, log_verbeux
from .utils import ajuster_dim


def fusion_ponderee(perception, prediction, confiance):
    """Combinaison CONTINUE, pondérée par confiance, de la perception réelle
    et de la génération prédite (§15.1) — jamais un remplacement binaire.
    `confiance` ∈ [0,1] : 1 = tout au prédit, 0 = tout au perçu. Bien
    distinct de la sélection de besoin (discrète, voir
    `priorisation_besoin_dominant`) — la fusion ici porte sur perception vs
    prédiction, jamais sur les besoins entre eux."""
    confiance = max(0.0, min(1.0, float(confiance)))
    perception = ajuster_dim(perception, prediction.numel())
    fusion = (1.0 - confiance) * perception + confiance * prediction
    log_verbeux("decision_action", "fusion_ponderee", confiance=confiance)
    return fusion


def recompense_intrinseque(L_total_avant, L_total_apres):
    """r_intrinsèque = L_total(t-1) - L_total(t) (§15.2) — unifie baisse
    d'erreur et baisse de complexité (L_total inclut le terme MDL, §3.3)."""
    return float(L_total_avant - L_total_apres)


def reflexe_cable(signal_danger, seuil=None):
    """Garde-fou NON APPRIS, JAMAIS atrophié — court-circuite toute
    sélection par besoin dominant (§15.3). Évalué EN PREMIER, toujours
    prioritaire sur k_t. `signal_danger` : scalaire (ex. douleur du
    `TableContexte`) ; retourne une commande de freinage, ou None si aucun
    danger."""
    seuil = seuil if seuil is not None else CONFIG["seuil_reflexe_douleur"]
    if signal_danger > seuil:
        log("decision_action", "reflexe_cable_declenche", signal_danger=signal_danger)
        return "freiner"
    return None


def generer_actions_candidates(actions_disponibles, table_besoins,
                               module_par_action=None, contexte=None):
    """Énumère les actions depuis 𝒜 (discret, borné), les évalue par
    rollout via `memoire_travail.FamilleHorizon`/`fusion_ponderee` (§15.3,
    M12). [Non résolu, marqué explicitement par la théorie elle-même,
    §15.3] : implémentation POC minimale qui réutilise strictement les
    mécanismes déjà posés, sans inventer de nouveau signal — affinable sans
    changer l'interface. Retourne {besoin: {action: valeur}} : u_k(a) =
    valeur prédite du besoin k à maturation."""
    scores = {besoin: {a: 0.0 for a in actions_disponibles} for besoin in table_besoins.etats}
    if module_par_action is None or contexte is None:
        return scores
    for action, module in module_par_action.items():
        if action not in actions_disponibles:
            continue
        with torch.no_grad():
            latent = module.forward_reconnaissance(contexte)
            prediction = module.forward_generation(latent)[: module.n_outputs_gen]
        valeur = -float(torch.mean(prediction ** 2))   # proxy : confiance de prédiction
        for besoin in scores:
            scores[besoin][action] = valeur
    return scores


def priorisation_besoin_dominant(table_besoins, actions_candidates, reflexe=None):
    """Sélectionne l'action selon le SEUL besoin actif k_t (§15.3,
    `TableBesoins.besoin_dominant`, argmax+hystérésis) — JAMAIS un mélange
    pondéré continu des besoins entre eux (c'est exactement ce que l'ancien
    `orchestrateur.souhaitabilite_torch` faisait, en violation du §0). Le
    garde-fou câblé, s'il est fourni (déjà déclenché), est prioritaire et
    court-circuite tout le reste."""
    if reflexe is not None:
        log("decision_action", "action_par_reflexe", action_choisie=reflexe)
        return reflexe
    k_t = table_besoins.besoin_dominant()
    candidates_k = actions_candidates.get(k_t, {})
    if not candidates_k:
        log_verbeux("decision_action", "priorisation_besoin_dominant",
                    besoin=k_t, action_choisie=None, raison="aucune_candidate")
        return None
    meilleure = max(candidates_k, key=candidates_k.get)
    log("decision_action", "priorisation_besoin_dominant", besoin=k_t, action_choisie=meilleure)
    return meilleure
