"""Boucle SCL — assemblage final (Phase 11) : cycle jour/nuit, persistance.

Chemin unique : l'ancien fork `CONFIG["pilotage_chantiers"]` et la couche
d'instrumentation "barreaux" (`evaluer_barreau1`, `calculer_barreaux`,
`deliberer_options`, `meilleure_projection`) sont abandonnés — ni l'un ni
l'autre n'appartenait à la spécification v6 ; le chemin structurel
(rupture → réparation → composition → création, §4.5) est le seul retenu.

`EtatSCL` rassemble tous les composants construits UNE FOIS par `main_loop`
(discriminateur, encodeur, décodeur, V_ψ...) et transportés explicitement —
jamais recréés en cours de route (§0)."""
import os
import time

import torch

from .attention import (
    AccumulateurOrchestrateur, PointerNetwork, SetTransformer, construire_T_t,
    entrainer_pointeurs, macro_pas,
)
from . import checkpoint as checkpoint_mod
from . import curiosite
from .config import CONFIG
from .credit import amorcage_creation, regret_composition, rejeu_contrefactuel_nocturne
from .disponibilite import logique_acceptation
from .dynamique import Dynamique
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
        self.dynamique = Dynamique()   # prédicteurs de dynamique du corps (émergents)
        self.jeu_apprentissage_gating = []
        self.trace_precedente = None


def sauvegarder_etat(chemin, etat):
    checkpoint_mod.sauvegarder(chemin, **etat.__dict__)


def charger_etat(chemin):
    composants = checkpoint_mod.charger(chemin)
    etat = EtatSCL.__new__(EtatSCL)
    etat.__dict__.update(composants)
    # compatibilité ascendante : un checkpoint sauvegardé avant l'ajout d'un
    # composant (ex. le modèle du corps) ne le contient pas — on le recrée pour
    # que la reprise multi-rounds d'un ancien cerveau.pkl ne casse jamais.
    defauts = {
        "dynamique": lambda: Dynamique(),
    }
    for nom, fabrique in defauts.items():
        if getattr(etat, nom, None) is None:
            setattr(etat, nom, fabrique())
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


# -------------------------------------------- sélection d'action par curiosité (§15.2)

def _incertitude_vision(g):
    m = g.modules.get("vision")
    return curiosite.incertitude(m) if m is not None else 0.0


def _action_curieuse(etat, monde, g):
    """Sélection d'action pilotée par la CURIOSITÉ (§4, §15.2) : choisir
    l'accélération dont la conséquence est la MOINS bien prédite (incertitude
    maximale = progrès d'apprentissage attendu maximal). Rester immobile (0,0)
    vaut en plus l'incertitude de la vision — rester → champ statique → la
    reconstruction progresse. Émergence : tant que la vision est incertaine,
    rester gagne (elle maîtrise le champ statique) ; une fois maîtrisée, seules
    les accélérations non encore maîtrisées restent incertaines → l'agent
    explore la dynamique de son corps, région par région. Aucune coordonnée
    d'objet, aucune géométrie de tâche."""
    v = (int(monde.vitesse[0]), int(monde.vitesse[1]))
    inc_vision = _incertitude_vision(g)
    valeurs = {}
    for accel in ACCELERATIONS_PERMISES:
        val = etat.dynamique.incertitude_action(v, accel)
        if accel == (0, 0):
            val = max(val, inc_vision)
        valeurs[accel] = val
    accel = max(valeurs, key=valeurs.get)
    log_verbeux("boucle", "action_curieuse", accel=list(accel),
                incertitude=round(valeurs[accel], 4), inc_vision=round(inc_vision, 4))
    return accel


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

    # 1. réflexe câblé — garde-fou inné, TOUJOURS prioritaire (§15.3)
    commande_reflexe = reflexe_frein(contexte_t["proprio"][:2], etat.table_contexte.douleur)

    # 2. sélection d'action par CURIOSITÉ (§15.2), hors réflexe : agir là où
    # l'incertitude prédictive est la plus grande. Aucune valuation géométrique,
    # aucune coordonnée d'objet — la compétence motrice émerge de la réduction
    # d'incertitude sur les conséquences des actions.
    if commande_reflexe is not None:
        commande = commande_reflexe
    else:
        commande = _action_curieuse(etat, monde, g)

    v_avant = (int(monde.vitesse[0]), int(monde.vitesse[1]))
    evenements = monde.appliquer_action(commande)
    etat.table_besoins.mettre_a_jour(evenements, vitesse_norme=vitesse_norme)
    etat.table_contexte.mettre_a_jour(evenements)

    # apprentissage de la DYNAMIQUE : la transition réelle (v_avant, accel) →
    # v_apres nourrit le prédicteur dédié à cette accélération, ou — si la
    # surprise vs le prior « rien ne change » est confirmée (SPRT) — en fait
    # NAÎTRE un (§4.5). C'est ici qu'émerge, région par région, la maîtrise du
    # corps.
    accel_exec = (int(monde.derniere_accel[0]), int(monde.derniere_accel[1]))
    v_apres = (int(monde.vitesse[0]), int(monde.vitesse[1]))
    etat.dynamique.observer(v_avant, accel_exec, v_apres, t=t)

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

    if t % CONFIG["periode_pouls"] == 0:
        _emettre_pouls(etat, monde, g, contexte_t, commande, t)
    log_verbeux("boucle", "pas_temps_reel", t=t, commande=commande,
               besoin_dominant=etat.table_besoins.besoin_dominant())
    return commande


def _emettre_pouls(etat, monde, g, contexte_t, commande, t):
    """Instantané compact de l'agent (non verbeux, cadence `periode_pouls`) —
    tout ce qu'il faut au dashboard pour montrer l'émergence : champ visuel VU
    vs PRÉVU (reconstruction), incertitude vision, incertitudes de dynamique
    par accélération, position/vitesse, action choisie. Un seul type
    d'événement à consommer côté viewer."""
    vision = g.modules.get("vision")
    champ_vu, champ_prevu = None, None
    if vision is not None and vision.dernier_latent is not None:
        c, h, w = vision.resolution
        with torch.no_grad():
            # chaînes compactes (les listes sont tronquées à 40 par le logger) :
            # dernière frame VUE vs dernière frame de la RECONSTRUCTION.
            vu = contexte_t["vision"][-1].reshape(-1).tolist()
            recon = vision.forward_generation(vision.dernier_latent)[: c * h * w]
            prevu = recon[-h * w:].tolist()
            champ_vu = ",".join(f"{x:.2f}" for x in vu)
            champ_prevu = ",".join(f"{float(x):.2f}" for x in prevu)
    dyn = {f"{a[0]},{a[1]}": round(etat.dynamique.incertitude_action(
              (int(monde.vitesse[0]), int(monde.vitesse[1])), a), 4)
           for a in ACCELERATIONS_PERMISES}
    rap = etat.dynamique.etat_maitrise()
    log("agent", "pouls", t=t,
        position=[int(monde.agent_pos[0]), int(monde.agent_pos[1])],
        vitesse=[int(monde.vitesse[0]), int(monde.vitesse[1])],
        action_choisie=[int(commande[0]), int(commande[1])],
        inc_vision=round(_incertitude_vision(g), 4),
        dyn_incertitude=dyn,
        n_predicteurs=len(etat.dynamique.predicteurs),
        n_maitrises=sum(1 for _, (_, m) in rap.items() if m),
        resolution=list(vision.resolution) if vision is not None else None,
        champ_vu=champ_vu, champ_prevu=champ_prevu)


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

def _etat_neuf(graine):
    graphe, discriminateur = construire_graphe_inne()
    return EtatSCL(graphe, discriminateur, Monde(graine=graine), TableBesoins())


def main_loop(n_jours=3, steps_par_jour=500, graine=None, verbose=False, checkpoint=None):
    """Boucle jour/nuit complète, persistance (§25, assemblage final)."""
    etat = None
    if checkpoint and checkpoint_mod.existe(checkpoint):
        try:
            etat = charger_etat(checkpoint)
        except Exception as exc:
            # checkpoint incompatible avec le code courant (ex. classe supprimée
            # entre deux versions) : la reprise ne doit JAMAIS casser. On archive
            # l'ancien cerveau et on repart d'un état neuf.
            secours = checkpoint + ".incompatible"
            try:
                os.replace(checkpoint, secours)
            except OSError:
                secours = "(non archivé)"
            log("boucle", "checkpoint_incompatible", chemin=checkpoint,
                erreur=str(exc), archive=secours)
            print(f"[SCL] checkpoint '{checkpoint}' incompatible avec le code actuel "
                  f"({type(exc).__name__}: {exc}). Ancien cerveau archivé → {secours}. "
                  f"Redémarrage d'un cerveau neuf.")
    if etat is None:
        etat = _etat_neuf(graine)

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
            dyn = etat.dynamique
            n_maitrises = sum(1 for _, (_, m) in dyn.etat_maitrise().items() if m)
            log("boucle", "resume_journee", jour=jour,
                sucres=etat.monde.compteurs["sucre"], batons=etat.monde.compteurs["baton"],
                erreur_globale=etat.graphe.erreur_globale(), n_modules=len(etat.graphe.modules),
                incertitude_vision=round(_incertitude_vision(etat.graphe), 4),
                n_predicteurs_dynamique=len(dyn.predicteurs),
                n_accels_maitrisees=n_maitrises)

    return etat
