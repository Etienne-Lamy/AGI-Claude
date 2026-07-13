"""Graphe SCL — structure pérenne de modules + opérations structurelles
(§1.2, §2, §4.6, §9).

Principe non négociable (§0) : les sorties d'un module sont DÉTACHÉES avant
d'alimenter le suivant — aucun gradient ne traverse une frontière de module.

`forward_graphe` est un TODO explicite de cette phase : sa forme cible
(§1.2, §10.3, §15.1) appelle `attention.macro_pas`, qui n'existe pas avant la
Phase 9. L'ancienne implémentation (gate/inhibition/fusion apprises dans
`orchestrateur.py`, désormais abandonné) violait le §0 (mélange continu,
`W_fusion` jamais entraîné) et n'est pas reprise : `Graphe` ne dépend plus
d'un orchestrateur du tout. Voir le stub plus bas pour le point d'ancrage
exact laissé pour la Phase 9/11.
"""
import math

import torch

from .config import CONFIG
from .logger import log, log_verbeux
from .module import Module, copier_module
from .simulateur import Simulateur
from .utils import ajuster_dim, kmeans2, separation_claire


def _quantile_normale(p):
    """Quantile (inverse CDF) de la loi normale standard, via erfinv."""
    p = min(max(p, 1e-9), 1 - 1e-9)
    p_t = torch.tensor(p, dtype=torch.float64)
    return float(torch.erfinv(2 * p_t - 1) * (2.0 ** 0.5))


def _quantile_chi2(df, p):
    """Quantile de la loi χ²_df — approximation de Wilson-Hilferty."""
    z = _quantile_normale(p)
    terme = max(1 - 2.0 / (9 * df) + z * (2.0 / (9 * df)) ** 0.5, 0.0)
    return df * terme ** 3


def test_non_inferiorite(erreurs_a, erreurs_b, delta=None, alpha=None):
    """Test de non-infériorité sur échantillons appariés (Blackwelder, 1982),
    §9 — primitive générique UNIQUE, réutilisée partout où une comparaison
    A/B doit être tranchée sans dégrader : verrouillage (plancher jamais
    plafond, §1.4), découpe, consolidation, certification d'un module créé.

    Convention : `erreurs_a` = référence (ancien/plancher), `erreurs_b` =
    candidat (nouveau) — plus petit est meilleur. B est déclaré NON INFÉRIEUR
    à A si son erreur ne dépasse pas celle de A de plus de `delta` (marge de
    dégradation tolérée), avec une confiance 1-α : on calcule la borne
    supérieure de l'IC unilatéral sur (erreur_b − erreur_a) et on accepte B
    ssi elle ne dépasse pas `delta`.

    Retourne True (accepter B) ou False (rejeter B, garder le plancher A)."""
    delta = delta if delta is not None else CONFIG["delta_non_inferiorite"]
    alpha = alpha if alpha is not None else CONFIG["alpha_non_inferiorite"]
    a = torch.as_tensor(erreurs_a, dtype=torch.float32).flatten()
    b = torch.as_tensor(erreurs_b, dtype=torch.float32).flatten()
    n = min(a.numel(), b.numel())
    if n == 0:
        return False   # rien à comparer : pas de preuve de non-infériorité
    d = b[:n] - a[:n]
    if n == 1:
        decision = bool(d.item() <= delta)
    else:
        moyenne = float(d.mean())
        erreur_std = float(d.std(unbiased=True)) / (n ** 0.5)
        borne_sup = moyenne + _quantile_normale(1 - alpha) * erreur_std
        decision = borne_sup <= delta
    log("graphe", "test_non_inferiorite", n=n, delta=delta, alpha=alpha,
        decision=decision)
    return decision


def croissance_gouvernee(phi_avant, phi_apres, epsilon=None):
    """Accepte une croissance ssi Φ_t décroît STRICTEMENT de plus de ε, à
    catalogue A_t fixé (§2.2) — monotonie par morceaux, pas de garantie
    globale au changement de support (cardinal de A_t)."""
    epsilon = epsilon if epsilon is not None else CONFIG["seuil_gain_croissance"]
    decision = (phi_avant - phi_apres) > epsilon
    log("graphe", "croissance_gouvernee", phi_avant=phi_avant, phi_apres=phi_apres,
        epsilon=epsilon, decision=decision)
    return decision


def rejet_gouverne(log_vraisemblance_h0, log_vraisemblance_h1, df, alpha=None):
    """Test du rapport de vraisemblance (Wilks, 1938) : G² = -2 log(L(H0)/L(H1))
    ~ χ²_df sous H0 (asymptotique) — conserve un module à gating conditionnel
    ssi son gate est significativement informatif ; sinon, rejette (§2.3,
    désuétude)."""
    alpha = alpha if alpha is not None else CONFIG["alpha_non_inferiorite"]
    g2 = -2.0 * (log_vraisemblance_h0 - log_vraisemblance_h1)
    seuil = _quantile_chi2(df, 1 - alpha)
    decision = "conserver" if g2 > seuil else "rejeter"
    log("graphe", "rejet_gouverne", g2=g2, seuil=seuil, df=df, decision=decision)
    return decision


class Graphe:
    def __init__(self):
        self.modules = {}
        self.edges = []            # (from_id, to_id, type)
        self.input_nodes = []
        self.output_nodes = []
        self.compositions = {}     # {noyau_id: {"amovible": id, "dim": int}}
        self.capteurs = {}         # {module_id: clé du flux sensoriel brut}
        self.erreurs_recentes = []
        self.simulateurs = {}      # {module_id: Simulateur} — création jumelée, §10.2

    # ------------------------------------------------------------ structure
    def ajouter_module(self, module, parents=(), enfants=(), input_node=False,
                       output_node=False):
        self.modules[module.id] = module
        for p in parents:
            self.edges.append((p, module.id, "standard"))
        for e in enfants:
            self.edges.append((module.id, e, "standard"))
        if input_node and module.id not in self.input_nodes:
            self.input_nodes.append(module.id)
        if output_node and module.id not in self.output_nodes:
            self.output_nodes.append(module.id)
        log("graphe", "ajout_module", module=module.id,
            parents=list(parents), enfants=list(enfants))

    def retirer(self, module_id):
        self.modules.pop(module_id, None)
        self.simulateurs.pop(module_id, None)
        self.edges = [e for e in self.edges if module_id not in (e[0], e[1])]
        for lst in (self.input_nodes, self.output_nodes):
            if module_id in lst:
                lst.remove(module_id)
        log("graphe", "retrait_module", module=module_id)

    def parents(self, module_id):
        return [f for f, t, _ in self.edges if t == module_id and f in self.modules]

    def enfants(self, module_id):
        return [t for f, t, _ in self.edges if f == module_id and t in self.modules]

    def ordre_topologique(self, restreindre=None):
        ids = [m for m in self.modules
               if restreindre is None or m in restreindre]
        resultat, marques = [], set()

        def visiter(mid):
            if mid in marques:
                return
            marques.add(mid)
            for p in self.parents(mid):
                if p in ids:
                    visiter(p)
            resultat.append(mid)

        for mid in ids:
            visiter(mid)
        return [m for m in resultat if m in ids]

    def erreur_globale(self):
        if not self.erreurs_recentes:
            return 0.0
        fen = self.erreurs_recentes[-CONFIG["fenetre_erreur_globale"]:]
        return sum(fen) / len(fen)

    def noter_erreur(self, e):
        self.erreurs_recentes.append(float(e))
        if len(self.erreurs_recentes) > 2 * CONFIG["fenetre_erreur_globale"]:
            del self.erreurs_recentes[: CONFIG["fenetre_erreur_globale"]]

    def entree_detachee(self, sources):
        """Concatène des sorties de modules PARENTS, détachées — aucun
        gradient ne traverse une frontière de module (§0). Point d'ancrage
        du détachement inter-module pour tout consommateur (aujourd'hui les
        tests ; demain `attention.macro_pas`, Phase 9)."""
        sources = [s.detach() if hasattr(s, "detach") else torch.as_tensor(s) for s in sources]
        return torch.cat(sources) if sources else torch.zeros(0)

    # ------------------------------------------------ passe avant (TODO Phase 9/11)
    def forward_graphe(self, contexte, mode="perception", restreindre=None):
        """Passe avant : perception / imagination / fusion, par macro-pas de
        largeur W (§1.2, §10.3, §15.1) — appelle `attention.macro_pas`.
        Non implémenté avant la Phase 9 (Set Transformer + Pointer Network) ;
        voir le docstring du module pour la justification de cet abandon de
        l'ancienne implémentation basée sur `orchestrateur.py`."""
        raise NotImplementedError(
            "forward_graphe dépend de attention.macro_pas — implémenté en Phase 9/11")

    # --------------------------------- Localisation du point de branchement, §4.6
    def localiser_point_branchement(self, contexte, seuil_pi_bas=None):
        """Premier module au sens du flux dont π_i(x) s'effondre alors que
        TOUS ses antécédents directs conservent π haut sur ce même contexte
        x (§4.6) — remplace `detecter_rupture` (algorithme différent, pas un
        renommage : balayage par antécédents sains, pas par seuil d'erreur
        global). Si le premier π effondré est un capteur (nœud d'entrée) :
        branchement en tête, retourne "capteur:<id>"."""
        seuil = seuil_pi_bas if seuil_pi_bas is not None else CONFIG["seuil_pi_bas"]
        for mid in self.ordre_topologique():
            m = self.modules[mid]
            if m.innate and m.locked_reco and m.locked_gen:
                continue   # réflexe câblé : jamais un point de branchement
            pi = m.fiabilite_contextuelle(contexte)
            if pi < seuil:
                parents = self.parents(mid)
                antecedents_sains = all(
                    self.modules[p].fiabilite_contextuelle(contexte) >= seuil
                    for p in parents if p in self.modules)
                if antecedents_sains:
                    if mid in self.input_nodes:
                        log("graphe", "localiser_point_branchement",
                            resultat="capteur", module=mid, pi=pi)
                        return f"capteur:{mid}"
                    log("graphe", "localiser_point_branchement", resultat=mid, pi=pi)
                    return mid
        log_verbeux("graphe", "localiser_point_branchement", resultat=None)
        return None

    # ------------------------------- création jumelée (§10.2, §4.5 étape 5)
    def creer_module_candidat(self, point_injection, n_inputs, n_latent,
                              contexte_echec=None, registre_rupture=None, t=0,
                              suffixe="candidat"):
        """Point d'entrée de la création jumelée : instancie SIMULTANÉMENT
        le module M_new (provisoire — §1.4, jamais de plancher certifié sur
        la seule base du rejeu simulé) et son simulateur associé S_new,
        ancré sur l'épisode fondateur réel `contexte_echec` s'il est fourni.
        Garde-fous : un seul candidat en cours, un seul par point de
        rupture, cooldown après abandon. Retourne (Module, Simulateur) ou
        None."""
        en_test = [m for m in self.modules.values() if m.status == "en_test"]
        if en_test:
            log("graphe", "creation_refusee", raison="candidat_deja_en_test",
                en_test=[m.id for m in en_test], point=point_injection)
            return None
        for m in self.modules.values():
            if m.point_rupture_origine == point_injection and m.status == "en_test":
                log("graphe", "creation_refusee", raison="point_deja_couvert",
                    point=point_injection)
                return None
        if registre_rupture is not None and not registre_rupture.peut_creer(
                point_injection, t):
            return None
        mid = f"{point_injection}_{suffixe}_{t}"
        # le candidat hérite du TYPE du module défaillant (conv reste conv)
        cls = (type(self.modules[point_injection])
               if point_injection in self.modules else Module)
        m = cls(mid, n_inputs_reco=n_inputs, n_latent=n_latent,
                point_rupture_origine=point_injection, status="en_test",
                provisoire=True)
        parents = self.parents(point_injection) if point_injection in self.modules else []
        enfants = self.enfants(point_injection) if point_injection in self.modules else []
        # un candidat à un nœud d'entrée doit recevoir le MÊME flux sensoriel
        # (sinon il naît orphelin : aucune entrée, aucun entraînement, jamais)
        est_capteur = point_injection in self.input_nodes
        self.ajouter_module(m, parents=parents, enfants=enfants,
                            input_node=est_capteur)
        if est_capteur:
            self.capteurs[mid] = self.capteurs.get(point_injection, point_injection)

        sim = Simulateur(f"{mid}_sim", dim_contexte_echec=n_inputs,
                         dim_latent_stocke=n_latent)
        if contexte_echec is not None:
            with torch.no_grad():
                z_fondateur = m.forward_reconnaissance(ajuster_dim(contexte_echec, n_inputs))
            sim.initialiser_depuis_episode(z_fondateur, contexte_echec)
        self.simulateurs[mid] = sim

        log("graphe", "creation_candidat", module=mid, simulateur=sim.id,
            point=point_injection, capteur=self.capteurs.get(mid), provisoire=True)
        return m, sim

    # ------------------------------------- Fragmentation (§9)
    def fragmenter_module(self, module, registre_cablage, contexte_t, t=0):
        """Module effondré → règle générale (dégelée) + exception (en test),
        avec competing_ids croisés."""
        # la règle générale CONSERVE l'identité du module d'origine :
        # cooldowns, tentatives en mémoire et câblage restent cohérents
        # (sinon le renommage relançait indéfiniment les mêmes opérations)
        regle = copier_module(module, module.id)
        regle.condensateur_reco = 0.3
        regle.condensateur_gen = 0.3
        regle.locked_reco = False
        regle.locked_gen = False
        regle.point_rupture_origine = module.point_rupture_origine
        exception = Module(f"{module.id}_exception_{t}",
                           n_inputs_reco=module.n_inputs_reco,
                           n_latent=module.n_latent,
                           n_outputs_gen=module.n_outputs_gen,
                           point_rupture_origine=module.id, status="en_test")
        regle.competing_ids = {exception.id}
        exception.competing_ids = {regle.id}
        registre_cablage.append(module_id=exception.id,
                                point_injection=module.id,
                                contexte=contexte_t,
                                signature_anomalie=module.friction_recente(),
                                t=t, type_="rupture")
        parents = self.parents(module.id)
        enfants = self.enfants(module.id)
        etait_input = module.id in self.input_nodes
        etait_output = module.id in self.output_nodes
        self.retirer(module.id)
        self.ajouter_module(regle, parents=parents, enfants=enfants,
                            input_node=etait_input, output_node=etait_output)
        self.ajouter_module(exception, parents=parents, enfants=enfants,
                            input_node=etait_input, output_node=etait_output)
        if etait_input:
            self.capteurs[exception.id] = self.capteurs.get(module.id, module.id)
        log("graphe", "fragmentation", origine=module.id,
            regle=regle.id, exception=exception.id)
        return regle, exception

    # ------------------------------------- Découpe (§9)
    def decouper_module(self, module, memoire_tampon, t=0):
        """Sépare noyau (copie exacte) + amovible additif sur la variable
        discriminante. Sortie composée = noyau(x) + gate × amovible(x[dim])."""
        hist = [(c, e) for c, e, _ in module.error_history if c is not None]
        if len(hist) < 20:
            log("graphe", "decoupe_impossible", module=module.id,
                raison="historique_insuffisant", n=len(hist))
            return None
        C = torch.stack([c for c, _ in hist])
        poids = torch.tensor([e for _, e in hist])
        labels, c0, c1 = kmeans2(C)
        # la découpe n'a de sens que si les erreurs diffèrent entre clusters
        e0 = float(poids[labels == 0].mean()) if (labels == 0).any() else 0.0
        e1 = float(poids[labels == 1].mean()) if (labels == 1).any() else 0.0
        if not separation_claire(C, labels, c0, c1) or abs(e0 - e1) < 0.05:
            log("graphe", "decoupe_impossible", module=module.id,
                raison="pas_de_separation_claire")
            return None
        variable_discriminante = int(torch.argmax(torch.abs(c0 - c1)))
        module.perf_avant_decoupe = module.friction_recente()

        # le noyau CONSERVE l'identité du module d'origine (copie exacte) —
        # le cooldown et les échecs de découpe suivent le module, pas son nom
        noyau = copier_module(module, module.id)
        noyau.point_rupture_origine = module.point_rupture_origine
        noyau.echecs_decoupe = getattr(module, "echecs_decoupe", 0)
        amovible = Module(f"{module.id}_amovible_{t}", n_inputs_reco=1,
                          n_latent=module.n_latent,
                          n_outputs_gen=module.n_latent,
                          point_rupture_origine=module.id, status="en_test")
        self.compositions[noyau.id] = {"amovible": amovible.id,
                                       "dim": variable_discriminante,
                                       "origine": module.id,
                                       "perf_avant": module.perf_avant_decoupe}
        parents = self.parents(module.id)
        enfants = self.enfants(module.id)
        etait_input = module.id in self.input_nodes
        etait_output = module.id in self.output_nodes
        self.retirer(module.id)
        self.ajouter_module(noyau, parents=parents, enfants=enfants,
                            input_node=etait_input, output_node=etait_output)
        # branche latérale : la relation additive noyau+amovible est portée par
        # self.compositions (pas d'arête amovible→noyau, qui polluerait
        # l'entrée du noyau en mode perception)
        self.ajouter_module(amovible, parents=parents)
        log("graphe", "decoupe", origine=module.id, noyau=noyau.id,
            amovible=amovible.id, variable_discriminante=variable_discriminante)
        return noyau, amovible

    def sortie_composee(self, noyau, amovible, x, gate=1.0, dim=None):
        """Combinaison additive noyau/amovible. gate=0 → sortie == noyau seul
        == module d'origine."""
        dim = dim if dim is not None else self.compositions.get(
            noyau.id, {}).get("dim", 0)
        # la dimension discriminante vient de l'espace du contexte : la
        # borner à l'espace d'entrée du module (tailles différentes)
        dim = min(int(dim), noyau.n_inputs_reco - 1)
        x = ajuster_dim(x, noyau.n_inputs_reco)
        base = noyau.forward_reconnaissance(x)
        if gate == 0.0:
            return base
        contrib = amovible.forward_reconnaissance(x[dim: dim + 1])
        return base + gate * ajuster_dim(contrib, base.numel())

    # ------------------------------------- Validation de découpe (§9)
    def valider_decoupe(self, noyau, amovible, memoire_tampon):
        """Décision intégrer / abandonner_amovible / fusionner_retour, via
        `test_non_inferiorite` réutilisé DEUX FOIS — même primitive, deux
        questions : (1) le noyau seul n'a-t-il pas régressé par rapport au
        module d'origine sur le groupe majoritaire ? (2) l'amovible
        améliore-t-il vraiment (pas juste "pas pire") le groupe minoritaire
        qu'il est censé capturer — testé avec une marge NÉGATIVE, qui exige
        une amélioration au-delà du bruit plutôt qu'une simple non-infériorité."""
        info = self.compositions.get(noyau.id, {})
        dim = min(int(info.get("dim", 0)), noyau.n_inputs_reco - 1)
        origine = info.get("origine", noyau.id)
        perf_avant = info.get("perf_avant", float("inf"))
        tentatives = [x for x in memoire_tampon.tentatives_reco
                      if x["module_id"] in (origine, noyau.id)]
        if not tentatives:
            log("graphe", "valider_decoupe", decision="abandonner_amovible",
                raison="aucune_tentative")
            return "abandonner_amovible"
        # partition par valeur de la variable discriminante (médiane)
        vals = torch.tensor([float(ajuster_dim(x["input"], noyau.n_inputs_reco)[dim])
                             for x in tentatives])
        mediane = float(vals.median())
        grp0 = [x for x, v in zip(tentatives, vals) if float(v) <= mediane]
        grp1 = [x for x, v in zip(tentatives, vals) if float(v) > mediane]
        majoritaire, minoritaire = (grp0, grp1) if len(grp0) >= len(grp1) else (grp1, grp0)

        erreurs_avant = [perf_avant] * max(len(majoritaire), 1)
        with torch.no_grad():
            erreurs_noyau = [
                float(torch.mean((ajuster_dim(x["cible"], noyau.n_latent)
                                  - noyau.forward_reconnaissance(x["input"])) ** 2))
                for x in majoritaire] or [float("inf")]
        noyau_non_inferieur = test_non_inferiorite(erreurs_avant, erreurs_noyau)

        with torch.no_grad():
            errs_sans, errs_avec = [], []
            for x in minoritaire:
                cible = ajuster_dim(x["cible"], noyau.n_latent)
                pred_sans = self.sortie_composee(noyau, amovible, x["input"], gate=0.0, dim=dim)
                pred_avec = self.sortie_composee(noyau, amovible, x["input"], gate=1.0, dim=dim)
                errs_sans.append(float(torch.mean((cible - pred_sans) ** 2)))
                errs_avec.append(float(torch.mean((cible - pred_avec) ** 2)))
            perf_amovible = sum(errs_avec) / len(errs_avec) if errs_avec else float("inf")

        if noyau_non_inferieur:
            amovible_ameliore = bool(errs_sans) and test_non_inferiorite(
                errs_sans, errs_avec, delta=-CONFIG["delta_non_inferiorite"])
            decision = "intégrer" if amovible_ameliore else "abandonner_amovible"
        else:
            decision = "fusionner_retour"
        log("graphe", "valider_decoupe", decision=decision,
            perf_amovible=perf_amovible, perf_avant=perf_avant,
            noyau_non_inferieur=noyau_non_inferieur)
        return decision

    # ------------------------------------- Atrophie (§2.3, §8.3 M6)
    def atrophier(self, module, registre_provenance=None):
        """Retire un module mûr n'ayant jamais acquis de certitude utile. Un
        module provisoire non confirmé qui atrophie voit ses exemples de
        provenance purgés en cascade (§8.3, M6) — la désuétude devient ainsi
        le mécanisme d'oubli des fausses croyances, sans mécanisme séparé."""
        if module.innate:
            return False
        if module.tentatives_count < CONFIG["maturite_structurelle"]:
            log_verbeux(module.id, "atrophie_ignoree_immature",
                        tentatives=module.tentatives_count)
            return False
        if (module.condensateur_reco < CONFIG["seuil_atrophie"]
                and module.condensateur_gen < CONFIG["seuil_atrophie"]):
            module.status = "abandonné"
            n_purges = registre_provenance.purger(module.id) if (
                module.provisoire and registre_provenance is not None) else 0
            log(module.id, "atrophie_abandon",
                condensateur_reco=module.condensateur_reco,
                condensateur_gen=module.condensateur_gen, n_purges=n_purges)
            return True
        return False

    # ------------------------------------- Contrôle de multiplicité (M10)
    def controle_multiplicite(self, tests_du_jour, alpha=None):
        """Applique le contrôle FDR (Benjamini-Hochberg) à l'ensemble des
        tests de non-infériorité exécutés dans la fenêtre du jour (M10) —
        réutilise `statistiques.controle_fdr`, ne réimplémente rien.
        `tests_du_jour` : liste de dicts porteurs d'une clé "p_valeur"."""
        from .statistiques import controle_fdr
        p_valeurs = [t["p_valeur"] for t in tests_du_jour]
        indices_acceptes, seuil = controle_fdr(p_valeurs, alpha=alpha)
        log("graphe", "controle_multiplicite", n=len(tests_du_jour),
            n_acceptes=len(indices_acceptes), seuil=seuil)
        return indices_acceptes, seuil

    # ------------------------------------- Recalage du plancher sous drift (M10)
    def recalage_plancher_drift(self, module, resultat_sprt):
        """Re-mesure le plancher c_i^min d'un module certifié si le SPRT de
        nouveauté (`statistiques.sprt_drift`) conclut à un drift durable sur
        son domaine (M10) — le verrouillage protège contre la régression à
        monde constant, pas contre le monde qui change."""
        decision, _ = resultat_sprt
        if decision == "H1":
            module.locked_reco = False
            module.locked_gen = False
            module.plancher_reco = None
            module.plancher_gen = None
            log(module.id, "recalage_plancher_drift", decision=decision)
            return True
        return False

    def committer_chemin(self, chemin, status="en_test"):
        """Ajoute les arêtes d'un chemin exploratoire validé en imagination."""
        nouvelles = []
        for a, b in zip(chemin, chemin[1:]):
            if not any(e[0] == a and e[1] == b for e in self.edges):
                self.edges.append((a, b, "exploratoire"))
                nouvelles.append((a, b))
        log("graphe", "committer_chemin", chemin=chemin,
            nouvelles_aretes=nouvelles, status=status)
        return nouvelles
