"""Mémoires de base SCL (§8) : besoins, contexte, tampon jour, exceptions,
registres structurels — disponibilité et provenance inclus (nouveaux en v6).
"""
from .config import CONFIG
from .logger import log, log_verbeux
from .utils import distance_contexte


class TableBesoins:
    """Vecteur de besoins b_t = (faim, ennui) — portée minimale à 2
    composantes (§1.1) ; même mécanisme, extensible à n>2 composantes plus
    tard sans refonte. `ennui` croît avec le temps écoulé depuis la dernière
    surprise validée (f croissante, f(0)=0, plafond dur à 0.5 — structurel,
    pas appris) ; `noter_surprise_validee` sera appelée par le pipeline SPRT
    (Phase 5) à chaque H1 confirmé.

    `besoin_dominant` : un seul besoin gouverne l'action à la fois, par
    argmax + hystérésis (Schmitt-trigger, §15.3) — jamais un mélange pondéré
    continu. C'est le mécanisme absent de l'ancien `orchestrateur.py`, qui
    mélangeait les besoins en continu via `souhaitabilite_torch` ; violation
    directe du §0/§15.3 de la v6, corrigée ici."""

    def __init__(self):
        self.etats = {"faim": 0.2, "ennui": 0.0}
        self._steps_depuis_surprise = 0
        self._dominant = None
        self.delta = CONFIG["delta_hysteresis_besoin"]

    def mettre_a_jour(self, evenements=(), vitesse_norme=0.0):
        """evenements : liste parmi {"sucre", "baton"} produite par le monde
        (seul "sucre" agit sur b_t ; "baton" est un signal de douleur câblé,
        géré par TableContexte, jamais mélangé aux besoins)."""
        self._steps_depuis_surprise += 1
        self.etats["faim"] = min(1.0, self.etats["faim"] + CONFIG["faim_par_step"]
                                 + CONFIG["faim_par_vitesse"] * vitesse_norme)
        for ev in evenements:
            if ev == "sucre":
                self.etats["faim"] = max(0.0, self.etats["faim"] - CONFIG["recompense_sucre"])
        self.etats["ennui"] = min(0.5, CONFIG["ennui_par_step"] * self._steps_depuis_surprise)
        if evenements:
            log("table_besoins", "mise_a_jour", etats=dict(self.etats),
                evenements=list(evenements))
        else:
            log_verbeux("table_besoins", "mise_a_jour", etats=dict(self.etats),
                        evenements=[])

    def noter_surprise_validee(self):
        """À appeler quand une surprise est validée (statistiques.sprt_surprise,
        Phase 5) : réinitialise l'horloge d'ennui — f(0) = 0."""
        self._steps_depuis_surprise = 0
        log("table_besoins", "surprise_validee")

    def besoin_dominant(self):
        """k_t : argmax + hystérésis de marge δ. Ne change de besoin dominant
        que si un autre dépasse le dominant courant de plus de δ ; sinon,
        conserve k_{t-1}."""
        cles = list(self.etats.keys())
        if self._dominant is None or self._dominant not in self.etats:
            self._dominant = max(cles, key=lambda k: self.etats[k])
            return self._dominant
        valeur_dominant = self.etats[self._dominant]
        meilleur = max(cles, key=lambda k: self.etats[k])
        if meilleur != self._dominant and self.etats[meilleur] > valeur_dominant + self.delta:
            ancien = self._dominant
            self._dominant = meilleur
            log("table_besoins", "besoin_dominant_change", ancien=ancien, nouveau=meilleur)
        return self._dominant


class TableContexte:
    """État de contexte global normal/choc — inhibe la création structurelle
    hors contexte normal (garde-fou de portée, §4). La douleur est un signal
    câblé indépendant de b_t (jamais mélangé aux besoins, §0) : réservé au
    réflexe câblé (`inne.reflexe_frein`) et à ce garde-fou. Décroît
    naturellement, monte sur collision avec un bâton."""

    def __init__(self, etat="normal"):
        self.etat = etat
        self.douleur = 0.0

    def mettre_a_jour(self, evenements=()):
        self.douleur = max(0.0, self.douleur - CONFIG["decroissance_douleur"])
        for ev in evenements:
            if ev == "baton":
                self.douleur = min(1.0, self.douleur + CONFIG["douleur_baton"])
        ancien = self.etat
        self.etat = "choc" if self.douleur > CONFIG["seuil_reflexe_douleur"] else "normal"
        if ancien != self.etat:
            log("table_contexte", "changement_etat", ancien=ancien, nouveau=self.etat,
                douleur=self.douleur)


class MemoireTampon:
    """Tentatives de la journée, rejouées pendant le cycle nocturne (§8.1)."""

    def __init__(self):
        self.tentatives_reco = []   # dicts: module_id, input, cible, erreur, t, contexte
        self.tentatives_gen = []

    def ajouter_reco(self, module_id, input_, cible, erreur, t, contexte=None):
        self.tentatives_reco.append(dict(
            module_id=module_id, input=input_.detach() if hasattr(input_, "detach") else input_,
            cible=cible.detach() if hasattr(cible, "detach") else cible,
            erreur=erreur, t=t, contexte=contexte))

    def ajouter_gen(self, module_id, latent, cible, erreur, t, contexte=None):
        self.tentatives_gen.append(dict(
            module_id=module_id, input=latent.detach() if hasattr(latent, "detach") else latent,
            cible=cible.detach() if hasattr(cible, "detach") else cible,
            erreur=erreur, t=t, contexte=contexte))

    def pour_point(self, point_ou_module_id):
        """Tentatives liées à un module (par id ou par point de rupture d'origine)."""
        r = [x for x in self.tentatives_reco if x["module_id"] == point_ou_module_id]
        g = [x for x in self.tentatives_gen if x["module_id"] == point_ou_module_id]
        return r, g

    def clear(self):
        n = len(self.tentatives_reco) + len(self.tentatives_gen)
        self.tentatives_reco.clear()
        self.tentatives_gen.clear()
        log("memoire_tampon", "clear", n_tentatives_effacees=n)


class MemoireExceptions:
    """Situations non résolues, revisitées la nuit (§4.5, esprit proche)."""

    def __init__(self):
        self.entrees = []  # dicts: contexte, erreur, resolved, dernier_essai

    def ajouter(self, contexte, erreur, t):
        self.entrees.append(dict(contexte=contexte, erreur=erreur,
                                 resolved=False, dernier_essai=t))
        log("memoire_exceptions", "ajout", erreur=erreur, t=t,
            n_total=len(self.entrees))

    def non_resolues(self):
        return [e for e in self.entrees if not e["resolved"]]


class RegistreCablage:
    """Historique des insertions structurelles (rupture / découpe / exploratoire
    / création) — base du rêve coordonné et de l'entraînement des pointeurs
    (§8, §9)."""

    def __init__(self):
        self.entrees = []

    def append(self, module_id, point_injection, contexte, signature_anomalie,
               t, type_):
        e = dict(module_id=module_id, point_injection=point_injection,
                 contexte=contexte, signature_anomalie=signature_anomalie,
                 t=t, type=type_, resultat=None, chemin=None)
        self.entrees.append(e)
        log("registre_cablage", "append", module_id=module_id,
            point_injection=point_injection, type=type_, t=t)
        return e


class RegistreRupture:
    """Cooldown de création par point de rupture (§1.4, esprit)."""

    def __init__(self):
        self.cooldown_creation = {}  # {point: t_dernier_abandon}

    def peut_creer(self, point, t):
        dernier = self.cooldown_creation.get(point, -float("inf"))
        ok = (t - dernier) > CONFIG["delai_refroidissement"]
        if not ok:
            log("registre_rupture", "creation_refusee_cooldown", point=point,
                t=t, dernier_abandon=dernier)
        return ok

    def marquer_abandon(self, point, t):
        self.cooldown_creation[point] = t
        log("registre_rupture", "abandon_marque", point=point, t=t)


class RegistreDisponibilite:
    """Échantillon varié 𝒲_i(t) par module, pour le test de disponibilité
    anticipée (§1.4 : plateau de progrès + stabilité du bruit résiduel).
    Dédoublonné par diversité — une stabilité apparente sur des contextes
    quasi identiques ne prouve rien — pas par consécutivité."""

    def __init__(self):
        self._echantillons = {}   # module_id -> liste de (contexte, erreur)

    def ajouter(self, module_id, contexte, erreur=None):
        w = self._echantillons.setdefault(module_id, [])
        c = contexte.detach() if hasattr(contexte, "detach") else contexte
        for c_existant, _ in w:
            if distance_contexte(c, c_existant) < CONFIG["seuil_diversite_disponibilite"]:
                return   # pas assez différent de ce qui est déjà échantillonné
        w.append((c, erreur))
        if len(w) > CONFIG["taille_echantillon_disponibilite"]:
            w.pop(0)
        log_verbeux("registre_disponibilite", "ajout", module_id=module_id, taille=len(w))

    def echantillon(self, module_id):
        return list(self._echantillons.get(module_id, []))


class RegistreProvenance:
    """Bit de provenance {réel, imaginé} attaché à chaque exemple stocké
    (§8.3, M6) — permet la purge en cascade des confabulations jamais
    confirmées (désuétude d'un module provisoire) et le plafonnement du
    ratio imaginé/réel d'un lot d'entraînement."""

    def __init__(self):
        self._par_module = {}   # module_id -> liste d'exemples tagués

    def marquer(self, exemple, provenance):
        assert provenance in ("reel", "imagine"), provenance
        exemple["provenance"] = provenance
        module_id = exemple.get("module_id")
        self._par_module.setdefault(module_id, []).append(exemple)
        log_verbeux("registre_provenance", "marquage", module_id=module_id,
                    provenance=provenance)
        return exemple

    def purger(self, module_id):
        n = len(self._par_module.pop(module_id, []))
        if n:
            log("registre_provenance", "purge", module_id=module_id, n_supprimes=n)
        return n

    def ratio_lot(self, lot):
        """Plafonne le ratio imaginé/réel d'un lot d'entraînement (Annexe B :
        3:1 par défaut). Sans exemple réel dans le lot, rien à plafonner
        contre une ancre réelle : renvoyé inchangé."""
        reels = [e for e in lot if e.get("provenance") == "reel"]
        imagines = [e for e in lot if e.get("provenance") == "imagine"]
        if not reels:
            return list(lot)
        plafond = int(len(reels) * CONFIG["plafond_ratio_imagine_reel"])
        if len(imagines) <= plafond:
            return list(lot)
        imagines_gardes = imagines[:plafond]
        log("registre_provenance", "ratio_plafonne", n_imagines_avant=len(imagines),
            n_imagines_apres=len(imagines_gardes), n_reels=len(reels))
        return reels + imagines_gardes
