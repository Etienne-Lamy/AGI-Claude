"""Module SCL — unité d'apprentissage strictement locale (E, G) : encodeur,
décodeur, condensateur, fiabilité contextuelle π(x), accumulateur de
gradient persistant ḡ, canal réinjecté, verrouillage asymétrique.

Principe non négociable (§0) : aucun gradient ne sort d'un module. Les
entrées inter-modules sont détachées par le graphe avant d'arriver ici ; les
cibles sont détachées ici. À l'intérieur d'un même module, en revanche, rien
n'interdit qu'un passage arrière traverse encodeur ET décodeur ensemble
(§2.1, BCD : Φ(θ)=Σ_i w_i L_i(θ_i^E,θ_i^G) — le module entier est un bloc,
pas nécessairement chaque sous-réseau) ; `entrainer_module_reco` et
`entrainer_module_gen` restent ici deux blocs BCD séparés (verrouillage
indépendant reco/gen), mais un module dérivé peut légitimement les
co-entraîner via une perte jointe (voir `module_visuel.entrainer_masque`).

Statut de verrouillage — TODO Phase 4 : le §1.4 exige "plancher, jamais
plafond" (une mise à jour ultérieure d'un module verrouillé doit être
acceptée si elle passe `graphe.test_non_inferiorite`, jamais bloquée par
principe). Cette phase pose le bookkeeping nécessaire (`plancher_reco`,
`plancher_gen`) mais garde encore le refus d'entraînement sur verrou comme
garde-fou temporaire, en attendant que `disponibilite.logique_acceptation`
(Phase 4) prenne le relais de cette décision.
"""
import math
import torch

from .config import CONFIG
from .logger import log, log_verbeux
from .utils import ajuster_dim, pente


def _param(a, b):
    return torch.nn.Parameter(torch.randn(a, b) * (1.0 / max(1, b) ** 0.5))


class Module:
    def __init__(self, id, n_inputs_reco, n_latent, n_outputs_gen=None,
                 n_hidden_reco=None, n_hidden_gen=None, innate=False,
                 point_rupture_origine=None, latent_input_dim=None,
                 is_action=False, status="actif", provisoire=False,
                 dim_reinjection=None):
        self.id = id
        self.innate = innate
        self.is_action = is_action
        self.point_rupture_origine = point_rupture_origine
        self.status = status  # "actif" | "en_test" | "abandonné" | "verrouillé"
        self.provisoire = provisoire   # §1.4 : jamais de plancher tant que non confirmé réel

        self.n_inputs_reco = n_inputs_reco
        self.n_latent = n_latent
        self.n_hidden_reco = n_hidden_reco or CONFIG["n_hidden_init"]
        self.n_hidden_gen = n_hidden_gen or CONFIG["n_hidden_init"]
        self.latent_input_dim = latent_input_dim or n_latent
        self.n_inputs_gen = self.latent_input_dim
        self.n_outputs_gen = n_outputs_gen if n_outputs_gen is not None else n_inputs_reco
        # sortie légèrement plus large que l'entrée (§1.2) : d_i' = d_i^cons + d_i^réinj
        self.dim_reinjection = (CONFIG["dim_reinjection"] if dim_reinjection is None
                                else dim_reinjection)
        self.n_sortie_decodeur = self.n_outputs_gen + self.dim_reinjection

        self._init_poids()
        self._rebuild_accumulateurs()

        self.condensateur_reco = 0.0
        self.condensateur_gen = 0.0
        self.locked_reco = False
        self.locked_gen = False
        self.plancher_reco = None      # meilleure erreur relative au verrouillage (§1.4)
        self.plancher_gen = None
        self.dernier_reinjecte = None  # dernier canal d_réinj produit (§1.2, §10.6)

        self.embedding = torch.randn(CONFIG["dim_emb"]) * 0.5  # espace latent COMMUN
        self.competing_ids = set()
        self.tentatives_count = 0
        self.meilleure_erreur_reco = float("inf")
        self.meilleure_erreur_gen = float("inf")
        self.error_history = []          # (contexte_vec | None, erreur, t)
        self.dernier_latent = None
        self.derniers_grads = [0.0, 0.0]  # [reco, gen]
        self.perf_avant_decoupe = None

        log(self.id, "creation_module", n_inputs_reco=n_inputs_reco,
            n_latent=n_latent, n_outputs_gen=self.n_outputs_gen,
            n_hidden_reco=self.n_hidden_reco, n_hidden_gen=self.n_hidden_gen,
            innate=innate, is_action=is_action, status=status,
            provisoire=provisoire, point_rupture_origine=point_rupture_origine)

    # ------------------------------------------------------------------ poids
    def _init_poids(self):
        self.encodeur_W1 = _param(self.n_hidden_reco, self.n_inputs_reco)
        self.encodeur_b1 = torch.nn.Parameter(torch.zeros(self.n_hidden_reco))
        self.encodeur_W2 = _param(self.n_latent, self.n_hidden_reco)
        self.encodeur_b2 = torch.nn.Parameter(torch.zeros(self.n_latent))
        self.decodeur_W1 = _param(self.n_hidden_gen, self.n_inputs_gen)
        self.decodeur_b1 = torch.nn.Parameter(torch.zeros(self.n_hidden_gen))
        self.decodeur_W2 = _param(self.n_sortie_decodeur, self.n_hidden_gen)
        self.decodeur_b2 = torch.nn.Parameter(torch.zeros(self.n_sortie_decodeur))

    def _rebuild_accumulateurs(self, voie=None):
        """(Re)initialise l'accumulateur de gradient persistant ḡ_i pour la
        voie concernée (§1.3) — appelé à la construction et après croissance
        dimensionnelle (les formes des paramètres changent)."""
        if voie in (None, "reco"):
            self._g_reco = [torch.zeros_like(p) for p in self.parametres_reco()]
        if voie in (None, "gen"):
            self._g_gen = [torch.zeros_like(p) for p in self.parametres_gen()]

    def parametres_reco(self):
        return [self.encodeur_W1, self.encodeur_b1, self.encodeur_W2, self.encodeur_b2]

    def parametres_gen(self):
        return [self.decodeur_W1, self.decodeur_b1, self.decodeur_W2, self.decodeur_b2]

    def parametres(self):
        return self.parametres_reco() + self.parametres_gen()

    def etat_dict(self):
        return {i: p.detach().clone() for i, p in enumerate(self.parametres())}

    def charger_etat(self, d):
        for i, p in enumerate(self.parametres()):
            with torch.no_grad():
                p.copy_(d[i])

    # -------------------------------------------- Fonction 4 : reconnaissance
    def forward_reconnaissance(self, input_niveau_inferieur):
        x = ajuster_dim(input_niveau_inferieur, self.n_inputs_reco)
        h = torch.relu(self.encodeur_W1 @ x + self.encodeur_b1)
        latent = self.encodeur_W2 @ h + self.encodeur_b2
        self.dernier_latent = latent.detach()
        log_verbeux(self.id, "forward_reconnaissance",
                    norme_input=float(x.detach().norm()),
                    norme_latent=float(latent.detach().norm()))
        return latent

    # ----------------------------------------------- Fonction 5 : génération
    def forward_generation(self, latent):
        """Renvoie la sortie COMPLÈTE : [d_i^cons (n_outputs_gen) | d_i^réinj
        (dim_reinjection)]. Les appelants qui ne consomment que la partie
        utile doivent trancher `sortie[:self.n_outputs_gen]`."""
        z = ajuster_dim(latent, self.n_inputs_gen)
        h = torch.relu(self.decodeur_W1 @ z + self.decodeur_b1)
        sortie_complete = self.decodeur_W2 @ h + self.decodeur_b2
        self.dernier_reinjecte = sortie_complete[self.n_outputs_gen:].detach()
        log_verbeux(self.id, "forward_generation",
                    norme_latent=float(z.detach().norm()),
                    norme_output=float(sortie_complete.detach().norm()))
        return sortie_complete

    # ------------------------------ Fonction 8 : aligner_action (poids figés)
    def aligner_action(self, projection_souhaitable, n_iterations=None):
        """Recherche sur le latent d'entrée UNIQUEMENT ; aucun poids modifié.
        Retourne la commande motrice (détachée, partie utile seulement)."""
        n_iterations = n_iterations or CONFIG["n_iterations_alignement"]
        cible = ajuster_dim(projection_souhaitable, self.n_outputs_gen).detach()
        etats = [(p, p.requires_grad) for p in self.parametres_gen()]
        for p, _ in etats:
            p.requires_grad_(False)
        latent = torch.randn(self.n_inputs_gen, requires_grad=True) * 0.1
        latent = latent.detach().requires_grad_(True)
        ecart_val = float("inf")
        for _ in range(n_iterations):
            sortie = self.forward_generation(latent)[: self.n_outputs_gen]
            ecart = torch.mean((sortie - cible) ** 2)
            (grad,) = torch.autograd.grad(ecart, latent)
            latent = (latent - CONFIG["lr_recherche_latent"] * grad).detach().requires_grad_(True)
            ecart_val = float(ecart.detach())
        for p, s in etats:
            p.requires_grad_(s)
        with torch.no_grad():
            commande = self.forward_generation(latent)[: self.n_outputs_gen].detach()
        log(self.id, "aligner_action", ecart_final=ecart_val,
            iterations=n_iterations, commande=commande)
        return commande

    # -------------- Cible prédictive de F9 (même mécanique de recherche que F8)
    def chercher_latent_predictif(self, input_prec, cible_future,
                                  n_iterations=None):
        """Le latent que l'encodeur AURAIT dû produire sur input_prec pour que
        le décodeur (figé) prédise cible_future. Recherche sur le latent seul,
        initialisée à enc(input_prec)."""
        n_iterations = n_iterations or CONFIG["n_iterations_latent_predictif"]
        cible = ajuster_dim(cible_future, self.n_outputs_gen).detach()
        etats = [(p, p.requires_grad) for p in self.parametres_gen()]
        for p, _ in etats:
            p.requires_grad_(False)
        with torch.no_grad():
            z0 = self.forward_reconnaissance(
                ajuster_dim(input_prec, self.n_inputs_reco).detach())
        z = ajuster_dim(z0, self.n_inputs_gen).detach().requires_grad_(True)
        for _ in range(n_iterations):
            sortie = self.forward_generation(z)[: self.n_outputs_gen]
            ecart = torch.mean((sortie - cible) ** 2)
            (grad,) = torch.autograd.grad(ecart, z)
            z = (z - CONFIG["lr_recherche_latent"] * grad).detach().requires_grad_(True)
        for p, s in etats:
            p.requires_grad_(s)
        log_verbeux(self.id, "latent_predictif", ecart_final=float(ecart.detach()))
        return z.detach()

    # ---------------------------- Accumulateur de gradient persistant (§1.3)
    def incorporer_gradient(self, grads, voie, phase="jour"):
        """ḡ_i ← β ḡ_i + (1-β)∇ — un seul état par module, cadence (β)
        différente jour/nuit. `grads` doit être aligné avec
        `parametres_reco()`/`parametres_gen()` selon `voie`."""
        beta = CONFIG["beta_jour"] if phase == "jour" else CONFIG["beta_nuit"]
        buf = self._g_reco if voie == "reco" else self._g_gen
        for i, g in enumerate(grads):
            if g is not None:
                buf[i].mul_(beta).add_(g, alpha=1 - beta)
        return buf

    def _appliquer_accumulateur(self, params, buf, lr):
        with torch.no_grad():
            for p, g in zip(params, buf):
                p -= lr * g

    # ------------------------------------- Fonction 9 : entraînement reco
    def entrainer_module_reco(self, input_observe, realite_attendue,
                              contexte_vec=None, t=0, phase="jour"):
        if self.locked_reco:
            # TODO Phase 4 : remplacer par disponibilite.logique_acceptation
            # (plancher jamais plafond) — refus temporaire en attendant.
            log(self.id, "entrainement_reco_refuse", raison="locked")
            return 0.0
        for p in self.parametres_reco():
            p.grad = None
        latent_predit = self.forward_reconnaissance(
            ajuster_dim(input_observe, self.n_inputs_reco).detach())
        realite = ajuster_dim(realite_attendue, self.n_latent).detach()
        erreur = torch.mean((realite - latent_predit) ** 2)
        erreur.backward()
        grads = [p.grad for p in self.parametres_reco()]
        grad_norme = math.sqrt(sum(float((g ** 2).sum()) for g in grads if g is not None))
        self.incorporer_gradient(grads, "reco", phase=phase)
        lr = CONFIG["lr_normal"] if self.condensateur_reco < 0.5 else CONFIG["lr_lent"]
        self._appliquer_accumulateur(self.parametres_reco(), self._g_reco, lr)
        e = float(erreur.detach())
        self.tentatives_count += 1
        self.meilleure_erreur_reco = min(self.meilleure_erreur_reco, e)
        self.derniers_grads[0] = grad_norme
        self._enregistrer_erreur(contexte_vec, e, t)
        log(self.id, "entrainement_reco", erreur=e, grad_norme=grad_norme, lr=lr, phase=phase)
        return e

    # -------------------------------------- Fonction 10 : entraînement gen
    def entrainer_module_gen(self, latent_source, realite_output,
                             contexte_vec=None, t=0, phase="jour"):
        if self.locked_gen:
            # TODO Phase 4 : idem entrainer_module_reco.
            log(self.id, "entrainement_gen_refuse", raison="locked")
            return 0.0
        for p in self.parametres_gen():
            p.grad = None
        output_predit = self.forward_generation(
            ajuster_dim(latent_source, self.n_inputs_gen).detach())[: self.n_outputs_gen]
        realite = ajuster_dim(realite_output, self.n_outputs_gen).detach()
        erreur = torch.mean((realite - output_predit) ** 2)
        erreur.backward()
        grads = [p.grad for p in self.parametres_gen()]
        grad_norme = math.sqrt(sum(float((g ** 2).sum()) for g in grads if g is not None))
        self.incorporer_gradient(grads, "gen", phase=phase)
        lr = CONFIG["lr_normal"] if self.condensateur_gen < 0.5 else CONFIG["lr_lent"]
        self._appliquer_accumulateur(self.parametres_gen(), self._g_gen, lr)
        e = float(erreur.detach())
        self.tentatives_count += 1
        self.meilleure_erreur_gen = min(self.meilleure_erreur_gen, e)
        self.derniers_grads[1] = grad_norme
        self._enregistrer_erreur(contexte_vec, e, t)
        log(self.id, "entrainement_gen", erreur=e, grad_norme=grad_norme, lr=lr, phase=phase)
        return e

    def entrainer_predictif(self, input_observe, cible_sortie, contexte_vec=None,
                            t=0, phase="jour"):
        """Entraînement conjoint reco+gen sur une paire (entrée → sortie) :
        input → E → latent → G → sortie, perte = MSE(sortie, cible). Un seul
        passage arrière à travers l'encodeur ET le décodeur du module (intra-
        module, autorisé §0/§2.1, comme `module_visuel.entrainer_masque`).
        Sert aux modules PRÉDICTEURS (ex. dynamique du corps : (v,accel) →
        v_suivant) créés dynamiquement. Enregistre l'erreur (→ incertitude,
        curiosite.py). Retourne l'erreur MSE."""
        if self.locked_reco and self.locked_gen:
            log(self.id, "entrainement_predictif_refuse", raison="locked")
            return 0.0
        params = self.parametres_reco() + self.parametres_gen()
        for p in params:
            p.grad = None
        latent = self.forward_reconnaissance(
            ajuster_dim(input_observe, self.n_inputs_reco).detach())
        sortie = self.forward_generation(latent)[: self.n_outputs_gen]
        realite = ajuster_dim(cible_sortie, self.n_outputs_gen).detach()
        erreur = torch.mean((realite - sortie) ** 2)
        erreur.backward()
        grads_reco = [p.grad for p in self.parametres_reco()]
        grads_gen = [p.grad for p in self.parametres_gen()]
        self.incorporer_gradient(grads_reco, "reco", phase=phase)
        self.incorporer_gradient(grads_gen, "gen", phase=phase)
        lr = CONFIG["lr_normal"] if self.condensateur_reco < 0.5 else CONFIG["lr_lent"]
        self._appliquer_accumulateur(self.parametres_reco(), self._g_reco, lr)
        self._appliquer_accumulateur(self.parametres_gen(), self._g_gen, lr)
        e = float(erreur.detach())
        self.tentatives_count += 1
        self.meilleure_erreur_reco = min(self.meilleure_erreur_reco, e)
        self.meilleure_erreur_gen = min(self.meilleure_erreur_gen, e)
        self._enregistrer_erreur(contexte_vec, e, t)
        log(self.id, "entrainement_predictif", erreur=e, phase=phase)
        return e

    def _enregistrer_erreur(self, contexte_vec, erreur, t):
        self.error_history.append((None if contexte_vec is None
                                   else contexte_vec.detach().clone(), erreur, t))
        if len(self.error_history) > CONFIG["taille_max_historique_erreurs"]:
            del self.error_history[: CONFIG["taille_max_historique_erreurs"] // 2]

    def friction_recente(self, fenetre=None):
        fenetre = fenetre or CONFIG["fenetre_friction"]
        if not self.error_history:
            return 0.0
        derniers = [e for _, e, _ in self.error_history[-fenetre:]]
        return sum(derniers) / len(derniers)

    def tendance_erreur(self, fenetre=50):
        return pente([e for _, e, _ in self.error_history[-fenetre:]])

    # ------------------------------ Fiabilité contextuelle π(x), §1.4 (Phase 2)
    def fiabilite_contextuelle(self, contexte_vec, k=5):
        """π(x) = 1 − L̂^relative(x), fiabilité indexée par CONTEXTE (pas
        par instant) : un module peut réussir en moyenne et échouer
        systématiquement sur une configuration précise — le condensateur
        global masque ce cas, π(x) le rend visible. Estimée par moyenne
        locale des k contextes les plus proches de l'historique d'erreurs,
        normalisée par le seuil de succès. Sans historique contextuel :
        valeur neutre (0.5), ni fiable ni non-fiable."""
        historique = [(c, e) for c, e, _ in self.error_history if c is not None]
        if contexte_vec is None or not historique:
            return 0.5
        cvec = ajuster_dim(contexte_vec, historique[0][0].numel()).detach()
        voisins = sorted(
            historique,
            key=lambda ce: float(((ajuster_dim(ce[0], cvec.numel()) - cvec) ** 2).sum()),
        )[:k]
        erreur_locale = sum(e for _, e in voisins) / len(voisins)
        relatif = min(CONFIG["cap_erreur_relative"],
                      erreur_locale / max(CONFIG["seuil_succes"], 1e-8))
        return max(0.0, 1.0 - relatif / CONFIG["cap_erreur_relative"])

    # ---------------------------------- Fonction 11 : condensateurs
    def mettre_a_jour_condensateurs(self, erreur_reco=None, erreur_gen=None,
                                    seuil_gen=None):
        s, ds, de = CONFIG["seuil_succes"], CONFIG["delta_succes"], CONFIG["delta_echec"]
        sg = seuil_gen if seuil_gen is not None else s
        if erreur_reco is not None and not self.locked_reco:
            if erreur_reco < s:
                self.condensateur_reco = min(1.0, self.condensateur_reco + ds)
            else:
                self.condensateur_reco = max(0.0, self.condensateur_reco - de)
        if erreur_gen is not None and not self.locked_gen:
            if erreur_gen < sg:
                self.condensateur_gen = min(1.0, self.condensateur_gen + ds)
            else:
                self.condensateur_gen = max(0.0, self.condensateur_gen - de)
        # Verrouillage indépendant reco / gen — jamais pour un module provisoire
        # (§1.4 : pas de plancher certifié sur la seule base du rejeu simulé).
        if (self.condensateur_reco >= CONFIG["seuil_verrou"] and not self.locked_reco
                and not self.provisoire):
            self.locked_reco = True
            self.plancher_reco = self.condensateur_reco
            log(self.id, "verrouillage", voie="reco", condensateur=self.condensateur_reco)
        if (self.condensateur_gen >= CONFIG["seuil_verrou"] and not self.locked_gen
                and not self.provisoire):
            self.locked_gen = True
            self.plancher_gen = self.condensateur_gen
            log(self.id, "verrouillage", voie="gen", condensateur=self.condensateur_gen)
        # journalisé seulement quand l'état change de façon visible (÷30 volume)
        etat_c = (int(self.condensateur_reco * 20), int(self.condensateur_gen * 20),
                  self.locked_reco, self.locked_gen)
        if etat_c != getattr(self, "_dernier_log_condensateurs", None):
            self._dernier_log_condensateurs = etat_c
            log(self.id, "condensateurs", reco=self.condensateur_reco,
                gen=self.condensateur_gen, locked_reco=self.locked_reco,
                locked_gen=self.locked_gen, provisoire=self.provisoire)

    def confirmer_reel(self):
        """Bascule le module de provisoire à confirmé — à appeler quand une
        occurrence RÉELLE (pas simulée via S_new) certifie le module (§1.4).
        Ne verrouille rien elle-même ; lève seulement le garde-fou qui
        empêchait le verrouillage."""
        if self.provisoire:
            self.provisoire = False
            log(self.id, "confirmation_reelle")

    # ---------------------------------- Fonction 12 : saturation
    def detecter_saturation(self, grad_norme_reco=None, grad_norme_gen=None,
                            erreur_reco=None, erreur_gen=None):
        """Double signal : gradient faible ET erreur mauvaise. Un module
        compétent (erreur basse) n'est jamais 'saturé'."""
        gr = self.derniers_grads[0] if grad_norme_reco is None else grad_norme_reco
        gg = self.derniers_grads[1] if grad_norme_gen is None else grad_norme_gen
        er = self.friction_recente() if erreur_reco is None else erreur_reco
        eg = self.friction_recente() if erreur_gen is None else erreur_gen
        sature_reco = (gr < CONFIG["seuil_grad_bas"]) and (er > CONFIG["seuil_erreur_mauvais"])
        sature_gen = (gg < CONFIG["seuil_grad_bas"]) and (eg > CONFIG["seuil_erreur_mauvais"])
        if sature_reco or sature_gen:
            log(self.id, "saturation_detectee", sature_reco=sature_reco,
                sature_gen=sature_gen, grad_reco=gr, grad_gen=gg,
                erreur_reco=er, erreur_gen=eg)
        return sature_reco, sature_gen

    # ---------------------------------- Croissance (support fonction 24)
    def grandir(self, voie, pas=None):
        """Ajoute des unités cachées en préservant la fonction existante
        (nouvelles lignes ~0 sur W2 : sortie inchangée au départ)."""
        pas = pas or CONFIG["croissance_pas"]
        if voie == "reco":
            if self.n_hidden_reco + pas > CONFIG["croissance_max"]:
                return False
            a = self.n_hidden_reco
            W1 = torch.cat([self.encodeur_W1.data,
                            torch.randn(pas, self.n_inputs_reco) * 0.01])
            b1 = torch.cat([self.encodeur_b1.data, torch.zeros(pas)])
            W2 = torch.cat([self.encodeur_W2.data,
                            torch.zeros(self.n_latent, pas)], dim=1)
            self.n_hidden_reco = a + pas
            self.encodeur_W1 = torch.nn.Parameter(W1)
            self.encodeur_b1 = torch.nn.Parameter(b1)
            self.encodeur_W2 = torch.nn.Parameter(W2)
        elif voie == "gen":
            if self.n_hidden_gen + pas > CONFIG["croissance_max"]:
                return False
            a = self.n_hidden_gen
            W1 = torch.cat([self.decodeur_W1.data,
                            torch.randn(pas, self.n_inputs_gen) * 0.01])
            b1 = torch.cat([self.decodeur_b1.data, torch.zeros(pas)])
            W2 = torch.cat([self.decodeur_W2.data,
                            torch.zeros(self.n_sortie_decodeur, pas)], dim=1)
            self.n_hidden_gen = a + pas
            self.decodeur_W1 = torch.nn.Parameter(W1)
            self.decodeur_b1 = torch.nn.Parameter(b1)
            self.decodeur_W2 = torch.nn.Parameter(W2)
        else:
            return False
        self._rebuild_accumulateurs(voie)
        log(self.id, "croissance", voie=voie, pas=pas,
            n_hidden_reco=self.n_hidden_reco, n_hidden_gen=self.n_hidden_gen)
        return True

    # ---------------------------------- Évaluations (sans apprentissage)
    def evaluer_reco(self, tentatives):
        if not tentatives:
            return float("inf")
        with torch.no_grad():
            errs = [float(torch.mean((ajuster_dim(x["cible"], self.n_latent)
                                      - self.forward_reconnaissance(x["input"])) ** 2))
                    for x in tentatives]
        return sum(errs) / len(errs)

    def evaluer_gen(self, tentatives):
        if not tentatives:
            return float("inf")
        with torch.no_grad():
            errs = [float(torch.mean((ajuster_dim(x["cible"], self.n_outputs_gen)
                                      - self.forward_generation(x["input"])[: self.n_outputs_gen]) ** 2))
                    for x in tentatives]
        return sum(errs) / len(errs)


def copier_module(module, nouveau_id):
    """Copie structurelle exacte (poids clonés, détachés). Préserve le TYPE
    du module (un module convolutif reste convolutif) et ses éventuels
    paramètres de construction spécialisés (ex. `resolution` pour un module
    visuel), repris génériquement via tout attribut du même nom qu'un
    argument de constructeur."""
    cls = type(module)
    kwargs = dict(n_outputs_gen=module.n_outputs_gen,
                  n_hidden_reco=module.n_hidden_reco,
                  n_hidden_gen=module.n_hidden_gen,
                  latent_input_dim=module.latent_input_dim,
                  is_action=module.is_action,
                  provisoire=module.provisoire,
                  dim_reinjection=module.dim_reinjection)
    if hasattr(module, "resolution"):
        kwargs["resolution"] = module.resolution
    m = cls(nouveau_id, module.n_inputs_reco, module.n_latent, **kwargs)
    m.charger_etat(module.etat_dict())
    m.innate = module.innate   # un module inné le reste à travers les copies
    m.embedding = module.embedding.detach().clone()
    m.condensateur_reco = module.condensateur_reco
    m.condensateur_gen = module.condensateur_gen
    m.locked_reco = module.locked_reco
    m.locked_gen = module.locked_gen
    m.plancher_reco = module.plancher_reco
    m.plancher_gen = module.plancher_gen
    m.error_history = list(module.error_history)
    m._g_reco = [g.clone() for g in module._g_reco]
    m._g_gen = [g.clone() for g in module._g_gen]
    log(nouveau_id, "copie_module", origine=module.id)
    return m
