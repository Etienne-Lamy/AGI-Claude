"""Attention SCL — le cœur de l'orchestrateur (§10, en entier).

Set Transformer en entrée (invariance/équivariance par permutation garantie
par construction, aucun encodage positionnel), Pointer Network en sortie
(triplet source/opérateur/cible par pointeurs softmax sur les indices
courants de T_t — jamais un vocabulaire fixe).

Chantier le plus dense et le plus risqué de la réécriture (construit et
validé en dernier parmi les mécanismes cognitifs, Phase 9, une fois toutes
ses dépendances — Phases 0-8 — testées indépendamment). Simplifications
POC, toutes explicitement sanctionnées par la théorie ou par le plan de
réécriture : vecteur de besoins à 2 composantes (Phase 1), une seule famille
d'horizon (Phase 7), V_ψ pouvant démarrer à 0 (Phase 8), catalogue
d'opérateurs minimal (`operateurs_natifs.py` différé, §25), et
`transfert_inter_dimensionnel` désactivé (no-op).
"""
import torch

from .config import CONFIG
from .logger import log, log_verbeux
from .utils import ajuster_dim


# --------------------------------------------------------- 1. Assemblage de T_t

def _type_module(module):
    """Typage minimal τ (§10.5) : suffisant pour le catalogue d'opérateurs
    POC, révisable sans changer l'interface."""
    if getattr(module, "is_action", False):
        return "action"
    return "latent"


def trace_autoreferentielle(trace_precedente):
    """Réinjecte trace_{t-1} (séquence de résultats de triplets émise au pas
    précédent) comme élément de F_t^ptr (§10.6) — application récursive de
    la définition de T_t à sa propre sortie passée, aucun mécanisme
    supplémentaire."""
    resultats = [ajuster_dim(pas["resultat"], CONFIG["dim_emb"])
                for pas in trace_precedente if pas.get("resultat") is not None]
    embedding = torch.stack(resultats).mean(dim=0) if resultats else torch.zeros(CONFIG["dim_emb"])
    return {"id": "trace_t-1", "embedding": embedding, "type": "trace"}


def construire_T_t(graphe, contexte, trace_precedente=None):
    """Assemble T_t = F_t^ctx ∪ F_t^ptr (§10.1, §1.2) à partir du graphe et
    du contexte courant. F_t^ptr : dernier latent de chaque module actif,
    le jeton [NEW], la trace auto-référentielle si fournie. F_t^ctx :
    condensateurs et fiabilités — jamais pointables, seulement consultés."""
    ptr, fiabilites = [], {}
    for mid, m in graphe.modules.items():
        if m.status not in ("actif", "verrouillé", "en_test"):
            continue
        pi = m.fiabilite_contextuelle(contexte)
        fiabilites[mid] = pi
        latent = m.dernier_latent if m.dernier_latent is not None else torch.zeros(m.n_latent)
        ptr.append({"id": mid, "embedding": ajuster_dim(latent, CONFIG["dim_emb"]),
                   "type": _type_module(m)})
    ptr.append({"id": "[NEW]", "embedding": torch.zeros(CONFIG["dim_emb"]), "type": "special"})
    if trace_precedente is not None:
        ptr.append(trace_autoreferentielle(trace_precedente))
    ctx = {
        "condensateurs": {mid: (m.condensateur_reco, m.condensateur_gen)
                          for mid, m in graphe.modules.items()},
        "fiabilites": fiabilites,
    }
    log_verbeux("attention", "construire_T_t", n_ptr=len(ptr))
    return {"ptr": ptr, "ctx": ctx}


# ---------------------------------------------------------------- 2. Set Transformer

class SetTransformer:
    """Encodeur Set Transformer minimal (Lee et al., 2019) : auto-attention
    multi-tête SANS encodage positionnel — l'invariance/équivariance par
    permutation est garantie par construction, rien dans l'architecture ne
    dépend de l'ordre des éléments."""

    def __init__(self, dim_entree=None, d_model=None, n_tetes=None):
        self.dim_entree = dim_entree or CONFIG["dim_emb"]
        self.d_model = d_model or CONFIG["d_model"]
        self.n_tetes = n_tetes or CONFIG["n_tetes_attention"]
        assert self.d_model % self.n_tetes == 0
        self.d_tete = self.d_model // self.n_tetes
        s = lambda a, b: torch.nn.Parameter(torch.randn(a, b) * (1.0 / max(1, b)) ** 0.5)
        self.W_in = s(self.d_model, self.dim_entree)
        self.W_q = s(self.d_model, self.d_model)
        self.W_k = s(self.d_model, self.d_model)
        self.W_v = s(self.d_model, self.d_model)
        self.W_out = s(self.d_model, self.d_model)
        self._g = [torch.zeros_like(p) for p in self.parametres()]

    def parametres(self):
        return [self.W_in, self.W_q, self.W_k, self.W_v, self.W_out]

    def encoder(self, embeddings):
        """embeddings : (N, dim_entree). Retourne (N, d_model), équivariant
        par permutation."""
        n = embeddings.shape[0]
        x = embeddings @ self.W_in.T
        Q = (x @ self.W_q.T).view(n, self.n_tetes, self.d_tete).transpose(0, 1)
        K = (x @ self.W_k.T).view(n, self.n_tetes, self.d_tete).transpose(0, 1)
        V = (x @ self.W_v.T).view(n, self.n_tetes, self.d_tete).transpose(0, 1)
        scores = (Q @ K.transpose(-2, -1)) / (self.d_tete ** 0.5)
        poids = torch.softmax(scores, dim=-1)
        sortie = (poids @ V).transpose(0, 1).reshape(n, self.d_model)
        return sortie @ self.W_out.T


# ------------------------------------------------------- 3. Pointer Network + masquage

def masque_compatibilite_type(logits, types, operateur):
    """Applique -∞ aux indices dont le type τ est incompatible avec
    l'opérateur pointé, AVANT softmax (§10.2, §10.5). `types_sortie=None`
    dans le catalogue (ex. "id") ⇒ compatible avec tout type."""
    compat = CONFIG["catalogue_operateurs"].get(operateur, {})
    types_autorises = compat.get("types_sortie")
    if types_autorises is None:
        return logits
    masque = torch.tensor([t in types_autorises for t in types], dtype=torch.bool)
    if not masque.any():
        return logits   # aucun candidat valable : ne pas produire un tenseur tout -inf
    return logits.masked_fill(~masque, float("-inf"))


def transfert_inter_dimensionnel(operateur, type_source, type_cible):
    """Autorise le transfert d'un opérateur appris sur un type vers un autre
    type partageant une propriété structurelle déclarée (§10.5). [H]
    hypothèse de structure analogique linéaire, non garantie ici —
    DÉSACTIVÉ (toujours refusé) pour le POC afin de réduire le rayon
    d'action du mécanisme le plus spéculatif de cette phase ; activable
    plus tard sans changer l'interface."""
    return False


def critere_arret_fil(fil):
    """Arrête un fil de décodage (§10.4) : incertitude propagée au-dessus
    d'un seuil, aucun candidat valable, port terminal atteint, ou
    profondeur maximale (garde-fou de dernier recours, jamais le critère
    principal). `fil` : dict de signaux accumulés par l'appelant."""
    if fil.get("port_terminal_atteint"):
        return True
    if fil.get("aucun_candidat_valable"):
        return True
    if fil.get("incertitude_cumulee", 0.0) > CONFIG["seuil_incertitude_fil"]:
        return True
    if fil.get("profondeur", 0) >= CONFIG["profondeur_max_fil"]:
        return True
    return False


class PointerNetwork:
    """Décodeur Pointer Network (Vinyals, Fortunato & Jaitly, 2015) : émet
    un triplet (source, opérateur, cible) par pointeurs softmax sur les
    indices COURANTS de T_t — jamais un vocabulaire de sortie fixe. Trois
    têtes de pointeur indépendantes (src/op/cib), chacune p(idx=j) =
    softmax_j(u^⊤tanh(W·k_j))."""

    def __init__(self, d_model=None, dim_op=None, operateurs=None):
        self.d_model = d_model or CONFIG["d_model"]
        self.dim_op = dim_op or CONFIG["dim_op"]
        self.operateurs = operateurs or list(CONFIG["catalogue_operateurs"])
        s = lambda a, b: torch.nn.Parameter(torch.randn(a, b) * (1.0 / max(1, b)) ** 0.5)
        v = lambda n: torch.nn.Parameter(torch.randn(n) * (1.0 / max(1, n)) ** 0.5)
        self.embeddings_op = s(len(self.operateurs), self.dim_op)
        self.W_src, self.u_src = s(self.d_model, self.d_model), v(self.d_model)
        self.W_op, self.u_op = s(self.dim_op, self.dim_op), v(self.dim_op)
        self.W_cib, self.u_cib = s(self.d_model, self.d_model), v(self.d_model)
        self._g = [torch.zeros_like(p) for p in self.parametres()]

    def parametres(self):
        return [self.embeddings_op, self.W_src, self.u_src, self.W_op, self.u_op,
                self.W_cib, self.u_cib]

    @staticmethod
    def _scores(W, u, keys):
        return torch.tanh(keys @ W.T) @ u

    def decoder(self, representation, elements):
        """representation : (N, d_model), alignée avec `elements` (liste de
        dicts "id"/"type"). Émet (triplet, log_prob) — [NEW] et "id" sont
        toujours des candidats valables (présents dans leurs catalogues
        respectifs). Masquage de type appliqué à la cible selon l'opérateur
        pointé (§10.2, §10.5)."""
        probs_src = torch.softmax(self._scores(self.W_src, self.u_src, representation), dim=0)
        idx_src = int(torch.multinomial(probs_src, 1))

        probs_op = torch.softmax(self._scores(self.W_op, self.u_op, self.embeddings_op), dim=0)
        idx_op = int(torch.multinomial(probs_op, 1))
        op = self.operateurs[idx_op]

        types = [e["type"] for e in elements]
        scores_cib = self._scores(self.W_cib, self.u_cib, representation)
        scores_cib = masque_compatibilite_type(scores_cib, types, op)
        probs_cib = torch.softmax(scores_cib, dim=0)
        idx_cib = int(torch.multinomial(probs_cib, 1))

        triplet = {"src": elements[idx_src]["id"], "op": op, "cib": elements[idx_cib]["id"]}
        log_prob = (torch.log(probs_src[idx_src] + 1e-12)
                   + torch.log(probs_op[idx_op] + 1e-12)
                   + torch.log(probs_cib[idx_cib] + 1e-12))
        return triplet, log_prob


# ---------------------------------------------------- 4. Exécution / macro-pas

def executer_triplet(graphe, triplet, elements_par_id):
    """Applique l'opérateur pointé sur la source, écrit le résultat à
    l'emplacement cible (§10.2, §10.3). "id" laisse la source inchangée
    (identité, toujours disponible). Une dépendance NON PRÊTE (source
    absente de `elements_par_id`, ou [NEW] non encore matérialisé) dégénère
    silencieusement en comportement id — bulle de pipeline, §10.3 — plutôt
    que d'échouer."""
    op, src_id, cib_id = triplet["op"], triplet["src"], triplet["cib"]
    element_src = elements_par_id.get(src_id)
    src_dispo = element_src is not None and element_src.get("embedding") is not None

    if op == "id" or not src_dispo or src_id == "[NEW]" or cib_id == "[NEW]":
        resultat = element_src["embedding"] if src_dispo else None
        log_verbeux("attention", "executer_triplet_id", src=src_id, cib=cib_id,
                    src_dispo=src_dispo)
        return resultat

    module_source = graphe.modules.get(src_id)
    if module_source is None or module_source.dernier_latent is None:
        return None
    with torch.no_grad():
        if op == "percevoir":
            resultat = module_source.forward_reconnaissance(module_source.dernier_latent)
        else:
            resultat = module_source.forward_generation(
                module_source.dernier_latent)[: module_source.n_outputs_gen]
    log_verbeux("attention", "executer_triplet", src=src_id, op=op, cib=cib_id)
    return resultat.detach()


def activation_creuse(A_t, w=None, cle_priorite=None):
    """Sélectionne les w≤W modules actifs parmi |A_t|≫W disponibles (§10.7)
    — codage parcimonieux sur catalogue surcomplet. `cle_priorite` : id →
    score (ex. π(x), condensateur) ; sans elle, ordre d'arrivée."""
    w = w if w is not None else CONFIG["W"]
    if cle_priorite is None:
        return list(A_t)[:w]
    return sorted(A_t, key=lambda i: -cle_priorite(i))[:w]


def macro_pas(graphe, T_t, encodeur, decodeur, allocation=None):
    """Lot de w≤W triplets simultanés (§10.3), tirés du MÊME encodage de
    T_t (simplification POC : les w triplets d'un macro-pas ne sont pas
    chaînés de façon autorégressive entre eux ; le chaînage inter-macro-pas
    passe par `trace_autoreferentielle`, réinjectée au macro-pas suivant)."""
    elements = T_t["ptr"]
    if not elements:
        return []
    embeddings = torch.stack([ajuster_dim(e["embedding"], CONFIG["dim_emb"]) for e in elements])
    representation = encodeur.encoder(embeddings)
    w = allocation if allocation is not None else min(CONFIG["W"], len(elements))
    elements_par_id = {e["id"]: e for e in elements}
    resultats = []
    for _ in range(w):
        triplet, log_prob = decodeur.decoder(representation, elements)
        resultat = executer_triplet(graphe, triplet, elements_par_id)
        resultats.append({"triplet": triplet, "log_prob": log_prob, "resultat": resultat})
    log_verbeux("attention", "macro_pas", w=w, n_elements=len(elements))
    return resultats


# --------------------------------------------------- 5. Apprentissage (REINFORCE)

def entrainer_pointeurs(encodeur, decodeur, trajectoire, phase="jour"):
    """REINFORCE (Williams, 1992) sur la trajectoire de triplets, baseline =
    regret de composition (§10.2, M8) — entraîne CONJOINTEMENT l'encodeur
    et le décodeur (même signal de politique traverse les deux ; l'un sans
    l'autre laisserait des gradients non consommés). `trajectoire` : liste
    de dicts {"log_prob", "regret"} (regret déjà calculé via
    `credit.regret_composition`, toujours ≥0, positif ⇒ choix sous-optimal
    ⇒ la perte pousse sa probabilité à la baisse)."""
    objets = (encodeur, decodeur)
    for obj in objets:
        for p in obj.parametres():
            p.grad = None
    perte = torch.zeros(())
    for pas in trajectoire:
        perte = perte + pas["log_prob"] * pas["regret"]
    perte = perte / max(len(trajectoire), 1)
    perte.backward()

    beta = CONFIG["beta_jour"] if phase == "jour" else CONFIG["beta_nuit"]
    for obj in objets:
        grads = [p.grad for p in obj.parametres()]
        for i, g in enumerate(grads):
            if g is not None:
                obj._g[i].mul_(beta).add_(g, alpha=1 - beta)
        with torch.no_grad():
            for p, g in zip(obj.parametres(), obj._g):
                p -= CONFIG["lr_pointeurs"] * g
    e = float(perte.detach())
    log("attention", "entrainement_pointeurs", perte=e, n_pas=len(trajectoire), phase=phase)
    return e


class AccumulateurOrchestrateur:
    """ḡ_orch (§10.8) : accumulateur de gradient PERSISTANT et DISTINCT de
    ḡ_i, alimenté par le résidu de pertinence de la composition (le regret
    de composition — la faute du CHOIX, pas la faute du module appelé).
    Même recette de moment que ḡ_i (Polyak, 1964 ; Kingma & Ba, 2014), état
    séparé."""

    def __init__(self):
        self.valeur = 0.0

    def mettre_a_jour(self, residu, phase="jour"):
        beta = CONFIG["beta_jour"] if phase == "jour" else CONFIG["beta_nuit"]
        self.valeur = beta * self.valeur + (1 - beta) * float(residu)
        log_verbeux("attention", "accumulateur_orchestrateur", valeur=self.valeur, phase=phase)
        return self.valeur
