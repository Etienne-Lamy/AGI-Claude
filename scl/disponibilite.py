"""Disponibilité anticipée et logique d'acceptation (§1.4).

Deux mécanismes qui remplacent l'heuristique de verrouillage ad hoc de
l'ancien code : `disponibilite_anticipee` décide QUAND un module rejoint
F_t^ptr (avant même le verrouillage complet) ; `logique_acceptation` décide
si une mise à jour d'un module DOIT être appliquée, sur la base de la
variation de sa fiabilité contextuelle plutôt que d'un simple refus binaire
une fois verrouillé (§0 : plancher, jamais plafond)."""
import torch

from .config import CONFIG
from .logger import log
from .utils import pente


def disponibilite_anticipee(module, registre_disponibilite,
                            epsilon_s=None, epsilon_sigma=None):
    """M_i disponible (avant verrouillage complet) ssi |ρ_i(t)| < ε_s
    (plateau de progrès, régression linéaire sur l'échantillon varié
    𝒲_i(t)) ET Var(σ̂_i) < ε_σ (stabilité du bruit résiduel).

    [D] σ̂_i est approximé ici par l'erreur enregistrée à chaque contexte de
    𝒲_i(t) (pas encore la tête hétéroscédastique dédiée de
    `statistiques.residu_normalise`, Phase 5) — simplification documentée :
    les deux tests portent sur la même série, ce qui reste cohérent avec
    l'intention (stabilité ET absence de progrès), à affiner plus tard."""
    epsilon_s = epsilon_s if epsilon_s is not None else CONFIG["epsilon_s"]
    epsilon_sigma = epsilon_sigma if epsilon_sigma is not None else CONFIG["epsilon_sigma"]
    echantillon = registre_disponibilite.echantillon(module.id)
    erreurs = [e for _, e in echantillon if e is not None]
    if len(erreurs) < CONFIG["taille_minimale_disponibilite"]:
        log(module.id, "disponibilite_anticipee", disponible=False,
            raison="echantillon_insuffisant", n=len(erreurs))
        return False
    rho = pente(erreurs)
    variance_bruit = (float(torch.var(torch.tensor(erreurs), unbiased=True))
                      if len(erreurs) > 1 else 0.0)
    disponible = abs(rho) < epsilon_s and variance_bruit < epsilon_sigma
    log(module.id, "disponibilite_anticipee", disponible=disponible,
        rho=rho, variance_bruit=variance_bruit, n=len(erreurs))
    return disponible


def logique_acceptation(module, x, y, phase="jour", voie="reco"):
    """Accepte/rejette une mise à jour de M_i sur l'exemple (x,y) selon la
    variation de π_i(x) (§1.4). La mise à jour est TOUJOURS tentée
    (incorporation à ḡ_i comprise) ; si π_i(x) augmente, elle est validée
    (condensateurs mis à jour). Si π_i(x) diminue, θ_i est restauré à l'état
    d'avant — mais la trace d'erreur de la tentative annulée reste dans
    `error_history` (déjà enregistrée par `entrainer_module_*`) : c'est elle
    qui annote le contexte comme peu fiable pour les futures π_i(x), sans
    mécanisme séparé de "prédicteur de fiabilité" à maintenir en plus."""
    pi_avant = module.fiabilite_contextuelle(x)
    params = module.parametres_reco() if voie == "reco" else module.parametres_gen()
    poids_avant = [p.detach().clone() for p in params]

    if voie == "reco":
        erreur = module.entrainer_module_reco(x, y, contexte_vec=x, phase=phase)
    else:
        erreur = module.entrainer_module_gen(x, y, contexte_vec=x, phase=phase)

    pi_apres = module.fiabilite_contextuelle(x)

    if pi_apres >= pi_avant:
        module.mettre_a_jour_condensateurs(
            **({"erreur_reco": erreur} if voie == "reco" else {"erreur_gen": erreur}))
        decision = "acceptee"
    else:
        with torch.no_grad():
            for p, ancien in zip(params, poids_avant):
                p.copy_(ancien)
        decision = "rejetee_contexte_signale"
    log(module.id, "logique_acceptation", decision=decision, voie=voie,
        pi_avant=pi_avant, pi_apres=pi_apres)
    return decision
