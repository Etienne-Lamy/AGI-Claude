"""Crédit SCL — décomposition du crédit entre le module appelé et le choix
de l'orchestrateur (§10.8, M7-M8) : regret de composition, approximation
journalière, rejeu contrefactuel nocturne à poids figés, amorçage à la
création."""
import torch

from .logger import log


def regret_composition(erreur_choisi, erreurs_candidats):
    """Regret = L̂_choisi(x,y) - min_j L̂_j(x,y) (§10.8, M7) — le rejeu
    contrefactuel (`rejeu_contrefactuel_nocturne`) sert de baseline. Module
    médiocre bien choisi ⇒ regret≈0, résidu module élevé. Bon module mal
    branché ⇒ regret élevé, résidu module bas — les deux fautes séparées."""
    if not erreurs_candidats:
        return 0.0
    regret = float(erreur_choisi - min(erreurs_candidats))
    log("credit", "regret_composition", regret=regret, n_candidats=len(erreurs_candidats))
    return regret


def approx_regret_jour(valeur_choisi, valeurs_candidats):
    """Approximation journalière du regret via V_ψ des alternatives NON
    prises, sans rejeu contrefactuel complet (coûteux) — §10.8(M7). Réemploi
    direct de `recherche.ValeurApprise`. Convention identique à
    `regret_composition` : positif ⇒ le choix était sous-optimal."""
    if not valeurs_candidats:
        return 0.0
    return float(max(valeurs_candidats) - valeur_choisi)


def rejeu_contrefactuel_nocturne(candidats, echantillon, voie="reco"):
    """Rejoue TOUS les candidats disponibles, poids FIGÉS (aucune mutation),
    contre la cible réelle, pour un échantillon de contextes (§8.1) —
    corrige le biais de sélection : on ne sait jamais ce qu'aurait donné
    l'alternative non choisie sans ce rejeu.

    `candidats` : dict {id: module}. `echantillon` : tentatives au format de
    `MemoireTampon` (dicts "input"/"cible"). Retourne
    {id_candidat: erreur_moyenne}. Vérifie explicitement qu'aucun poids n'a
    bougé (garde-fou, pas seulement une conséquence attendue)."""
    residus = {}
    for cid, module in candidats.items():
        poids_avant = [p.detach().clone() for p in module.parametres()]
        fn = module.evaluer_reco if voie == "reco" else module.evaluer_gen
        erreur = fn(echantillon)
        poids_apres = module.parametres()
        assert all(torch.equal(a, p.detach()) for a, p in zip(poids_avant, poids_apres)), (
            f"rejeu_contrefactuel_nocturne : poids de {cid} modifiés — violation §8.1")
        residus[cid] = erreur
    log("credit", "rejeu_contrefactuel_nocturne", n_candidats=len(candidats),
        n_echantillon=len(echantillon), voie=voie)
    return residus


def amorcage_creation(module_cree, contexte_creation, jeu_apprentissage_gating):
    """Injecte immédiatement (x_création, M_i, positif) dans le jeu
    d'apprentissage du gating (§8.2) — pas d'attente passive du rejeu
    nocturne pour la première association : un module neuf a un embedding
    non façonné, sans amorçage rien ne pousse l'attention à le retrouver."""
    exemple = {"contexte": contexte_creation, "module_id": module_cree.id, "label": "positif"}
    jeu_apprentissage_gating.append(exemple)
    log(module_cree.id, "amorcage_creation",
        contexte_present=contexte_creation is not None)
    return exemple
