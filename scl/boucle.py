"""Boucle SCL — assemblage final (Phase 11) : cycle jour/nuit, persistance.

Chemin unique : l'ancien fork `CONFIG["pilotage_chantiers"]` et la couche
d'instrumentation "barreaux" (`evaluer_barreau1`, `calculer_barreaux`,
`deliberer_options`, `meilleure_projection`) sont abandonnés — ni l'un ni
l'autre n'appartenait à la spécification v6 ; le chemin structurel
(rupture → réparation → composition → création, §4.5) est le seul retenu.

`EtatSCL` rassemble tous les composants construits UNE FOIS par `main_loop`
(discriminateur, encodeur, décodeur, V_ψ...) et transportés explicitement —
jamais recréés en cours de route (§0)."""
import time

import torch

from .attention import (
    AccumulateurOrchestrateur, PointerNetwork, SetTransformer, construire_T_t,
    entrainer_pointeurs, macro_pas,
)
from . import checkpoint as checkpoint_mod
from .config import CONFIG
from .credit import amorcage_creation, regret_composition, rejeu_contrefactuel_nocturne
from .decision_action import priorisation_besoin_dominant
from .disponibilite import logique_acceptation
from .inne import construire_graphe_inne, reflexe_frein
from .logger import log, log_verbeux, set_temps
from .memoires import (
    MemoireExceptions, MemoireTampon, RegistreCablage, RegistreDisponibilite,
    RegistreProvenance, RegistreRupture, TableBesoins, TableContexte,
)
from .monde import ACCELERATIONS_PERMISES, Monde
from .recherche import ValeurApprise, a_etoile_ancree
from .statistiques import residu_module, sprt_creation, sprt_drift
from .utils import ajuster_dim


class EtatSCL:
    """Conteneur de l'état complet transporté par la boucle jour/nuit."""

    def __init__(self, graphe, discriminateur, monde, table_besoins):
        self.graphe = graphe
        self.discriminateur = discriminateur
        self.monde = monde
        self.table_besoins = table_besoins
        self.table_contexte = TableContexte()
        self.memoire_tampon = MemoireTampon()
        self.memoire_exceptions = MemoireExceptions()
        self.registre_cablage = RegistreCablage()
        self.registre_rupture = RegistreRupture()
        self.registre_disponibilite = RegistreDisponibilite()
        self.registre_provenance = RegistreProvenance()
        self.encodeur = SetTransformer()
        self.decodeur = PointerNetwork()
        self.valeur_apprise = ValeurApprise()
        self.accumulateur_orchestrateur = AccumulateurOrchestrateur()
        self.jeu_apprentissage_gating = []
        self.trace_precedente = None


def sauvegarder_etat(chemin, etat):
    checkpoint_mod.sauvegarder(chemin, **etat.__dict__)


def charger_etat(chemin):
    composants = checkpoint_mod.charger(chemin)
    etat = EtatSCL.__new__(EtatSCL)
    etat.__dict__.update(composants)
    return etat


# ------------------------------------------------------------- contexte / capteurs

def construire_contexte_enrichi(contexte_t, table_besoins, erreur_globale):
    """Résumé fixe du contexte (composante F_t^ctx) : vision moyennée par
    ligne, proprioception, besoins, erreur globale — un vecteur de taille
    `CONFIG["dim_contexte"]`."""
    vision = torch.as_tensor(contexte_t["vision"][-1], dtype=torch.float32)
    vision_pool = vision.mean(dim=0)
    proprio = torch.as_tensor(contexte_t["proprio"], dtype=torch.float32)
    besoins_vec = torch.tensor(list(table_besoins.etats.values()), dtype=torch.float32)
    brut = torch.cat([vision_pool, proprio, besoins_vec, torch.tensor([float(erreur_globale)])])
    return ajuster_dim(brut, CONFIG["dim_contexte"])


def inputs_bruts(contexte_t):
    """Capteurs bruts en tenseurs (composante physique de F_t^ptr)."""
    return {
        "vision": torch.as_tensor(contexte_t["vision"][-1], dtype=torch.float32),
        "proprio": torch.as_tensor(contexte_t["proprio"], dtype=torch.float32),
    }


# --------------------------------------------------------- sélection d'action (§15)

def _cellules_traversees(vx, vy):
    """Cellules relatives franchies en un pas à la vitesse (vx, vy) — miroir
    exact de la détection de collision de `monde.appliquer_action` (division
    entière plancher), pour anticiper de la même façon un contact bâton."""
    n_pas = int(max(abs(vx), abs(vy), 1))
    return [((vx * i) // n_pas, (vy * i) // n_pas) for i in range(1, n_pas + 1)]


def _predire_vitesse(v, accel, v_max):
    return (max(-v_max, min(v_max, int(v[0]) + accel[0])),
            max(-v_max, min(v_max, int(v[1]) + accel[1])))


def _penalite_baton(v_prime, batons_set):
    """Pénalité si le pas prédit franchit un bâton (évitement, pas seulement
    freinage réflexe une fois le choc subi)."""
    if not batons_set:
        return 0.0
    for c in _cellules_traversees(v_prime[0], v_prime[1]):
        if c in batons_set:
            return CONFIG["penalite_baton_navigation"]
    return 0.0


def _scores_actions(monde, besoins):
    """Génération d'actions candidates (§15.3) par rollout à horizon 1 avec un
    modèle physique du corps : chaque accélération est évaluée sur la position
    PRÉDITE au pas suivant (v' = clip(v + accel)), pas sur l'alignement de
    l'accélération seule. Conséquence clé : la vitesse courante est prise en
    compte — l'agent freine pour atterrir sur le sucre au lieu de le survoler
    (gestion de la vitesse, cœur du problème). Le modèle « v' → position »
    est ici la vérité-terrain du corps ; il constitue le point de
    branchement où un modèle APPRIS (module de prévision) pourra le remplacer
    sans changer l'interface (§15.3, migration vers l'apprentissage)."""
    v = monde.vitesse
    v_max = monde.v_max
    sucres, batons = monde.objets_visibles()
    batons_set = {(int(a), int(b)) for a, b in batons}
    scores_faim, scores_ennui = {}, {}
    for accel in ACCELERATIONS_PERMISES:
        v_prime = _predire_vitesse(v, accel, v_max)
        penalite = _penalite_baton(v_prime, batons_set)

        # faim : minimiser la distance au sucre APRÈS déplacement (le sucre en
        # (di,dj) devient (di-v'x, dj-v'y)) — atterrissage, pas survol.
        if sucres:
            dist_apres = min(((di - v_prime[0]) ** 2 + (dj - v_prime[1]) ** 2) ** 0.5
                             for di, dj in sucres)
            scores_faim[accel] = -dist_apres - penalite
        else:
            # aucun sucre en vue : explorer en gardant de l'élan (évite de se figer)
            scores_faim[accel] = 0.3 * (abs(v_prime[0]) + abs(v_prime[1])) - penalite

        # ennui : exploration — récompense le déplacement, évite les bâtons
        scores_ennui[accel] = (abs(v_prime[0]) + abs(v_prime[1])) - penalite

    return {"faim": scores_faim, "ennui": scores_ennui}


# ------------------------------------------------------------- entraînement local

def _entrainer_localement(etat, contexte_t, inputs, ctx_vec, commande_choisie, t):
    g = etat.graphe
    champ = inputs["vision"]

    if "vision" in g.modules:
        m = g.modules["vision"]
        e = m.entrainer_masque(champ, t=t)
        m.mettre_a_jour_condensateurs(erreur_reco=e, erreur_gen=e)
        etat.registre_disponibilite.ajouter("vision", ctx_vec, e)
        etat.memoire_tampon.ajouter_reco("vision", champ, champ, e, t, contexte=ctx_vec)
        g.noter_erreur(e)

    if "proprio" in g.modules:
        m = g.modules["proprio"]
        decision = logique_acceptation(m, m.dernier_latent, inputs["proprio"], voie="gen")
        etat.registre_disponibilite.ajouter("proprio", ctx_vec, m.friction_recente())
        log_verbeux("boucle", "entrainement_proprio", decision=decision)

    if "integration" in g.modules and "vision" in g.modules and "proprio" in g.modules:
        m = g.modules["integration"]
        cible = g.entree_detachee([g.modules["vision"].dernier_latent,
                                   g.modules["proprio"].dernier_latent])
        e = m.entrainer_module_reco(cible, m.dernier_latent if m.dernier_latent is not None
                                    else torch.zeros(m.n_latent), contexte_vec=ctx_vec, t=t)
        m.mettre_a_jour_condensateurs(erreur_reco=e)
        etat.registre_disponibilite.ajouter("integration", ctx_vec, e)

    if "action" in g.modules and "integration" in g.modules:
        m = g.modules["action"]
        latent_int = g.modules["integration"].dernier_latent
        if latent_int is not None:
            cible = torch.tensor([float(commande_choisie[0]), float(commande_choisie[1])])
            decision = logique_acceptation(m, latent_int, cible, voie="gen")
            log_verbeux("boucle", "entrainement_action", decision=decision)


# --------------------------------------------- recherche de composition (§7.4-7.5)

def _tenter_composition(g, point_reel, valeur_apprise):
    """Recherche A* ancrée (§7.5) d'un module déjà certifié pouvant couvrir
    le point de rupture, tentée avant toute création quand le simulacre est
    jugé plausible (§4.5 étape 3). Voisinage = graphe existant (parents et
    enfants directs) ; ancrage = tout module verrouillé rencontré en chemin
    (§7.4, point de vérité = module certifié). Sur ce POC, le graphe est
    quasi linéaire : l'échec de recherche est attendu la plupart du temps,
    pas un bug (dégénérescence exhaustive documentée, `recherche.py`)."""

    def voisins(mid):
        if mid not in g.modules:
            return []
        return [(v, 1.0) for v in list(g.parents(mid)) + list(g.enfants(mid))
                if v in g.modules]

    def ancres(mid):
        m = g.modules.get(mid)
        if m is not None and m.locked_reco:
            return "intermediaire", mid
        return None

    def objectif(mid):
        return mid != point_reel and g.modules[mid].locked_reco

    chemin = a_etoile_ancree(point_reel, objectif, voisins, valeur_apprise, ancres)
    return chemin if chemin and len(chemin) > 1 else None


# -------------------------------------------------------------- pas de temps réel

def boucle_temps_reel(etat, t=0):
    """Pas de temps réel complet (§4.5, §10, §15) : perception, garde-fou
    câblé (évalué en premier), composition (attention), sélection et
    exécution de l'action, apprentissage local, vérification structurelle
    périodique."""
    g, monde = etat.graphe, etat.monde
    contexte_t = monde.percevoir()
    inputs = inputs_bruts(contexte_t)
    vitesse_norme = float(torch.as_tensor(contexte_t["proprio"][:2], dtype=torch.float32).norm())

    # perception : peuple dernier_latent des modules d'entrée (sans grad, ce
    # n'est pas ici qu'on apprend — l'entraînement a ses propres forwards)
    with torch.no_grad():
        if "vision" in g.modules:
            g.modules["vision"].forward_reconnaissance(inputs["vision"])
        if "proprio" in g.modules:
            g.modules["proprio"].forward_reconnaissance(inputs["proprio"])
        if "integration" in g.modules and "vision" in g.modules and "proprio" in g.modules:
            entree_int = g.entree_detachee([g.modules["vision"].dernier_latent,
                                            g.modules["proprio"].dernier_latent])
            g.modules["integration"].forward_reconnaissance(entree_int)

    erreur_globale = g.erreur_globale()
    ctx_vec = construire_contexte_enrichi(contexte_t, etat.table_besoins, erreur_globale)

    # composition (attention) — exercée et entraînée à chaque pas, même si
    # elle ne détermine pas encore directement la commande motrice de ce POC
    # (§10.2 : l'orchestrateur compose, il ne calcule jamais lui-même)
    T_t = construire_T_t(g, ctx_vec, trace_precedente=etat.trace_precedente)
    trajectoire = macro_pas(g, T_t, etat.encodeur, etat.decodeur,
                            allocation=min(CONFIG["W"], len(T_t["ptr"])))
    etat.trace_precedente = trajectoire

    # 1. réflexe câblé — TOUJOURS évalué en premier, prioritaire (§15.3)
    commande_reflexe = reflexe_frein(contexte_t["proprio"][:2], etat.table_contexte.douleur)

    # 2. sélection d'action par besoin dominant (jamais un mélange, §0/§15.3)
    scores = _scores_actions(monde, etat.table_besoins)
    commande = priorisation_besoin_dominant(etat.table_besoins, scores, reflexe=commande_reflexe)
    if commande is None:
        commande = (0, 0)

    evenements = monde.appliquer_action(commande)
    etat.table_besoins.mettre_a_jour(evenements, vitesse_norme=vitesse_norme)
    etat.table_contexte.mettre_a_jour(evenements)

    _entrainer_localement(etat, contexte_t, inputs, ctx_vec, commande, t)

    # 3. vérification structurelle périodique (§4.5 : localisation → réparation
    # accumulée → SPRT de création → plausibilité → composition ou création)
    if t % CONFIG["periode_verification_structurelle"] == 0 and etat.table_contexte.etat == "normal":
        point = g.localiser_point_branchement(ctx_vec)
        if point is not None:
            point_reel = point.split("capteur:", 1)[-1] if point.startswith("capteur:") else point
            if point_reel in g.modules:
                m_defaillant = g.modules[point_reel]
                cible = ajuster_dim(ctx_vec, m_defaillant.n_outputs_gen)
                residu = residu_module(m_defaillant, ctx_vec, cible)
                etat.registre_rupture.enregistrer_echec(point_reel, ctx_vec, residu)

                decision, _ = sprt_creation(
                    etat.registre_rupture.echecs_pour(point_reel),
                    d=m_defaillant.n_outputs_gen)
                log_verbeux("boucle", "sprt_creation_point", point=point_reel,
                           decision=decision)

                if decision in ("H0", "H1"):
                    etat.registre_rupture.purger_echecs(point_reel)

                if decision == "H1":
                    # module manquant confirmé (échecs distincts accumulés,
                    # pas un incident isolé) — teste la plausibilité avant
                    # de créer : simulacre plausible → composition d'abord
                    plausible = etat.discriminateur.evaluer_plausibilite(ctx_vec)
                    chemin = None
                    if plausible >= CONFIG["seuil_plausibilite"]:
                        chemin = _tenter_composition(g, point_reel, etat.valeur_apprise)
                    if chemin is not None:
                        log("boucle", "composition_trouvee", point=point_reel,
                            chemin=chemin)
                        etat.registre_rupture.marquer_abandon(point_reel, t)
                    else:
                        resultat = g.creer_module_candidat(
                            point_reel, n_inputs=m_defaillant.n_inputs_reco,
                            n_latent=m_defaillant.n_latent, contexte_echec=ctx_vec,
                            registre_rupture=etat.registre_rupture, t=t)
                        if resultat is not None:
                            nouveau_module, simulateur = resultat
                            etat.registre_cablage.append(
                                module_id=nouveau_module.id, point_injection=point_reel,
                                contexte=ctx_vec, signature_anomalie="rupture", t=t, type_="rupture")
                            amorcage_creation(nouveau_module, ctx_vec, etat.jeu_apprentissage_gating)

    log_verbeux("boucle", "pas_temps_reel", t=t, commande=commande,
               besoin_dominant=etat.table_besoins.besoin_dominant())
    return commande


# ------------------------------------------------------------------ cycle nocturne

def reve_coordonne(etat):
    """Entraînement contrastif nocturne de D_φ à partir du câblage de la
    journée (§8.3, esprit) : contexte réel (positif) contre une variante
    perturbée (négatif) — garde le discriminateur calibré sur ce qui a
    réellement structuré le graphe."""
    entrees = etat.registre_cablage.entrees[-CONFIG["n_reve_coordonne"]:]
    n = 0
    for entree in entrees:
        contexte = entree.get("contexte")
        if contexte is None:
            continue
        positif = ajuster_dim(contexte, etat.discriminateur.dimension)
        negatifs = [positif + torch.randn_like(positif) * 2.0
                   for _ in range(CONFIG["n_contrefactuels"])]
        etat.discriminateur.entrainer_contrastif(positif, negatifs, phase="nuit")
        n += 1
    log("boucle", "reve_coordonne", n_entrees=n)


def cycle_nocturne(etat, t=0):
    """Pipeline nocturne complet (§4.5, §7.2, §8, §9, M1, M6, M10) : rejeu
    contrefactuel → REINFORCE des pointeurs, recalage sous drift, atrophie
    (purge de provenance), rêve coordonné, purge du tampon jour."""
    g = etat.graphe

    # 1. rejeu contrefactuel -> regret -> REINFORCE des pointeurs
    candidats = {mid: m for mid, m in g.modules.items() if not m.innate}
    echantillon = etat.memoire_tampon.tentatives_reco[-50:]
    if candidats and echantillon:
        residus = rejeu_contrefactuel_nocturne(candidats, echantillon)
        if etat.trace_precedente:
            trajectoire_regret = []
            for pas in etat.trace_precedente:
                src = pas["triplet"]["src"]
                if src in residus:
                    regret = regret_composition(residus[src], list(residus.values()))
                    trajectoire_regret.append({"log_prob": pas["log_prob"], "regret": regret})
                    etat.accumulateur_orchestrateur.mettre_a_jour(regret, phase="nuit")
            if trajectoire_regret:
                entrainer_pointeurs(etat.encodeur, etat.decodeur, trajectoire_regret, phase="nuit")

    # 2. recalage du plancher sous drift, pour les modules verrouillés
    fen = CONFIG["taille_fenetre_drift"]
    for m in g.modules.values():
        if m.locked_reco and not m.innate:
            recents = [e for _, e, _ in m.error_history[-fen:]]
            anciens = [e for _, e, _ in m.error_history[-2 * fen:-fen]]
            if recents and anciens:
                resultat = sprt_drift(anciens, recents)
                g.recalage_plancher_drift(m, resultat)

    # 3. atrophie (purge de provenance en cascade pour les modules provisoires)
    for m in list(g.modules.values()):
        g.atrophier(m, registre_provenance=etat.registre_provenance)

    # 4. rêve coordonné (discriminateur)
    reve_coordonne(etat)

    # 5. purge du tampon jour
    etat.memoire_tampon.clear()
    log("boucle", "cycle_nocturne_termine", n_modules=len(g.modules))


# --------------------------------------------------------------------- boucle globale

def main_loop(n_jours=3, steps_par_jour=500, graine=None, verbose=False, checkpoint=None):
    """Boucle jour/nuit complète, persistance (§25, assemblage final)."""
    if checkpoint and checkpoint_mod.existe(checkpoint):
        etat = charger_etat(checkpoint)
    else:
        graphe, discriminateur = construire_graphe_inne()
        monde = Monde(graine=graine)
        table_besoins = TableBesoins()
        etat = EtatSCL(graphe, discriminateur, monde, table_besoins)

    for jour in range(n_jours):
        set_temps(jour=jour)
        for step in range(steps_par_jour):
            set_temps(step=step)
            boucle_temps_reel(etat, t=step)
            if CONFIG["delai_step"] > 0:
                time.sleep(CONFIG["delai_step"])
        cycle_nocturne(etat, t=steps_par_jour)
        if checkpoint:
            sauvegarder_etat(checkpoint, etat)
        if verbose:
            log("boucle", "resume_journee", jour=jour,
                sucres=etat.monde.compteurs["sucre"], batons=etat.monde.compteurs["baton"],
                erreur_globale=etat.graphe.erreur_globale(), n_modules=len(etat.graphe.modules))

    return etat.graphe, etat.monde, etat.table_besoins
