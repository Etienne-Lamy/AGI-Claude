"""Allocation dynamique de la capacité d'attention (§13) — budget total W
partagé entre fils/rôles concurrents (perception, prédiction, création,
délibération), proportionnellement à leur urgence (Weighted Fair Queueing,
Demers-Keshav-Shenker 1989 ; capacité attentionnelle limitée, Kahneman 1973)."""
from .config import CONFIG
from .logger import log, log_verbeux


def urgence_fil(nom_fil, table_besoins, residu_surprise=0.0,
                poids_besoin=1.0, poids_surprise=1.0):
    """Urgence d'un fil dérivée des signaux de besoin (intensité du besoin
    DOMINANT, §15.3) et de surprise (résidu déjà calculé par
    `statistiques.sprt_surprise`) — réemploi strict, aucun nouveau signal
    (§13). Les poids relatifs des deux ingrédients sont au choix de
    l'appelant (un fil "création" est piloté par la surprise, un fil
    "perception" plutôt par le besoin)."""
    dominant = table_besoins.besoin_dominant()
    intensite_besoin = table_besoins.etats.get(dominant, 0.0)
    urgence = poids_besoin * intensite_besoin + poids_surprise * max(residu_surprise, 0.0)
    urgence = max(urgence, CONFIG["urgence_plancher"])
    log_verbeux("allocation_attention", "urgence_fil", fil=nom_fil, urgence=urgence)
    return urgence


def allouer_capacite(urgences, W=None):
    """w_k(t) = ⌊W · u_k(t)/Σ_j u_j(t)⌋, Σ_k w_k(t) ≤ W (§13, WFQ) —
    allocation proportionnelle sous ressource bornée. `urgences` :
    {nom_fil: u_k(t)}. Urgence totale nulle ⇒ répartition égale (dégénère
    proprement, ne divise jamais par zéro)."""
    W = W if W is not None else CONFIG["W"]
    if not urgences:
        return {}
    total = sum(urgences.values())
    if total <= 0:
        part = W // len(urgences)
        allocation = {k: part for k in urgences}
    else:
        allocation = {k: int((W * u) / total) for k, u in urgences.items()}
    log_verbeux("allocation_attention", "allouer_capacite", urgences=urgences,
                allocation=allocation, W=W)
    return allocation


def role_creation(table_besoins, residu_surprise, phase="jour", W=None):
    """La création de module est un fil concurrent EXPLICITE (§13, M5),
    servi par WFQ comme tout autre rôle — sous le garde-fou câblé, qui reste
    prioritaire absolu (géré en amont par `decision_action.reflexe_cable`,
    pas ici). Partage jour/nuit : jour = minimum viable (capter le
    contexte, amorcer E/G et S_new) ; nuit = part pleine (entraînement
    complet via S_new, réemploi ḡ_i)."""
    W = W if W is not None else CONFIG["W"]
    urgence = urgence_fil("creation", table_besoins, residu_surprise=residu_surprise,
                          poids_besoin=0.0, poids_surprise=1.0)
    if phase == "jour":
        part = min(CONFIG["part_creation_jour_min"], W)
    else:
        allocation = allouer_capacite({"creation": urgence, "reste": max(W - 1, 1)}, W=W)
        part = allocation.get("creation", 1)
    log("allocation_attention", "role_creation", phase=phase, urgence=urgence, part=part)
    return part
