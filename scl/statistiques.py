"""Statistiques SCL — résidu normalisé, trois usages du SPRT générique
(surprise, création, drift), contrôle FDR, cadence variable (§4, M1, M10).

Toutes les instances de SPRT ici réutilisent `utils.sprt_sequentiel` — même
statistique, trois contextes d'application différents, comme l'exige le §4.3
("réemploi §4.3, 3e usage" etc. dans l'architecture cible)."""
import torch

from .config import CONFIG
from .logger import log
from .utils import ajuster_dim, distance_contexte, sprt_sequentiel


def _increment_gaussien(x, mu0, mu1, var):
    """log(p1(x)/p0(x)) pour x ~ N(·, var), comparant deux hypothèses de
    moyenne (mu0 sous H0, mu1 sous H1) — approximation gaussienne du résidu
    de Mahalanobis (χ²_d ≈ N(d, 2d) par TCL, valable dès que d n'est pas
    minuscule)."""
    return ((x - mu0) ** 2 - (x - mu1) ** 2) / (2.0 * var)


def _variance(valeurs):
    n = len(valeurs)
    if n < 2:
        return 0.0
    m = sum(valeurs) / n
    return sum((v - m) ** 2 for v in valeurs) / (n - 1)


# ------------------------------------------------------------- résidu normalisé

def residu_normalise(x, mu, sigma):
    """S(x) = (x-μ)^⊤ Σ^{-1} (x-μ) ~ χ²_d sous H0 (§4.2). Σ ISOTROPE ici
    (Σ=σ²I, un scalaire par prédiction) — simplification documentée en
    attendant une tête hétéroscédastique pleinement covariante (Simulateur,
    Phase 6) ; d est implicitement la dimension de x."""
    x = torch.as_tensor(x, dtype=torch.float32).flatten()
    mu = ajuster_dim(mu, x.numel())
    sigma = max(float(sigma), 1e-6)
    ecart = x - mu
    return float((ecart @ ecart) / (sigma ** 2))


def residu_module(module, contexte, cible):
    """Résidu normalisé d'un module sur (contexte, cible) : μ = prédiction
    ponctuelle du module (décodeur), σ² = variance récente de son erreur
    (proxy isotrope de Σ_θ)."""
    with torch.no_grad():
        latent = module.forward_reconnaissance(contexte)
        mu = module.forward_generation(latent)[: module.n_outputs_gen]
    erreurs_recentes = [e for _, e, _ in module.error_history[-20:]] or [1.0]
    sigma2 = max(sum(erreurs_recentes) / len(erreurs_recentes), 1e-6)
    return residu_normalise(cible, mu, sigma2 ** 0.5)


# ------------------------------------------------------------------- SPRT × 3

def sprt_surprise(flux_residus, d, alpha=None, beta=None, decalage=None):
    """SPRT sur le résidu associé à π_i(x) (§4.1–4.3) — 1er usage. H0 :
    résidu conforme à χ²_d (approx. N(d,2d)) ; H1 : résidu translaté
    (surprise). Retourne (décision, n)."""
    alpha = alpha if alpha is not None else CONFIG["alpha_sprt_surprise"]
    beta = beta if beta is not None else CONFIG["beta_sprt_surprise"]
    decalage = decalage if decalage is not None else CONFIG["decalage_sprt_surprise"] * d
    mu0, var = float(d), float(2 * d)
    increments = [_increment_gaussien(r, mu0, mu0 + decalage, var) for r in flux_residus]
    decision, n = sprt_sequentiel(increments, alpha, beta)
    log("statistiques", "sprt_surprise", decision=decision, n=n, d=d)
    return decision, n


def sprt_creation(echecs, d, alpha=None, beta=None, decalage=None):
    """SPRT sur les échecs de réparation successifs (M1) — 2e usage.
    `echecs` : liste de (contexte, résidu). Contextes DISTINCTS exigés
    (dédoublonnage par diversité, réemploi §1.4) : un seul incident isolé ne
    déclenche jamais la création. Franchissement de H1 ⇒ 'module manquant'."""
    alpha = alpha if alpha is not None else CONFIG["alpha_sprt_creation"]
    beta = beta if beta is not None else CONFIG["beta_sprt_creation"]
    decalage = decalage if decalage is not None else CONFIG["decalage_sprt_surprise"] * d
    contextes_distincts, residus = [], []
    for contexte, residu in echecs:
        if all(distance_contexte(contexte, c) >= CONFIG["seuil_diversite_disponibilite"]
               for c in contextes_distincts):
            contextes_distincts.append(contexte)
            residus.append(residu)
    mu0, var = float(d), float(2 * d)
    increments = [_increment_gaussien(r, mu0, mu0 + decalage, var) for r in residus]
    decision, n = sprt_sequentiel(increments, alpha, beta)
    log("statistiques", "sprt_creation", decision=decision, n=n,
        n_echecs_bruts=len(echecs), n_contextes_distincts=len(residus))
    return decision, n


def sprt_drift(residus_anciens, residus_recents, alpha=None, beta=None, decalage=None):
    """SPRT de nouveauté appliqué au domaine d'un module certifié (M10) —
    3e usage. H0 : les résidus récents suivent la même distribution que les
    résidus anciens (déjà certifiés) ; H1 : translation (drift durable)."""
    alpha = alpha if alpha is not None else CONFIG["alpha_sprt_drift"]
    beta = beta if beta is not None else CONFIG["beta_sprt_drift"]
    if not residus_anciens:
        return "continuer", 0
    mu0 = sum(residus_anciens) / len(residus_anciens)
    var = max(_variance(residus_anciens), 1e-6)
    decalage = decalage if decalage is not None else CONFIG["decalage_sprt_drift"] * (var ** 0.5)
    increments = [_increment_gaussien(r, mu0, mu0 + decalage, var) for r in residus_recents]
    decision, n = sprt_sequentiel(increments, alpha, beta)
    log("statistiques", "sprt_drift", decision=decision, n=n, mu0=mu0, var=var)
    return decision, n


# ------------------------------------------------------------- contrôle FDR

def controle_fdr(p_valeurs, alpha=None):
    """Contrôle du taux de fausses découvertes (Benjamini & Hochberg, 1995)
    sur l'ensemble des tests exécutés dans la fenêtre du jour (M10). Retourne
    (indices_acceptes, seuil_effectif) : les indices dont on rejette H0
    (résultat déclaré significatif/accepté) après correction — plus grand
    rang k tel que p_(k) ≤ (k/m)·α, tous les rangs ≤ k acceptés."""
    alpha = alpha if alpha is not None else CONFIG["alpha_non_inferiorite"]
    m = len(p_valeurs)
    if m == 0:
        return [], 0.0
    ordre = sorted(range(m), key=lambda i: p_valeurs[i])
    indices_acceptes, seuil_effectif = [], 0.0
    for rang, i in enumerate(ordre, start=1):
        seuil_k = (rang / m) * alpha
        if p_valeurs[i] <= seuil_k:
            indices_acceptes = ordre[:rang]
            seuil_effectif = seuil_k
    log("statistiques", "controle_fdr", m=m, n_acceptes=len(indices_acceptes),
        seuil_effectif=seuil_effectif)
    return indices_acceptes, seuil_effectif


# --------------------------------------------------------------- cadence variable

def cadence_variable(type_flux):
    """Cadence d'échantillonnage du SPRT selon l'échelle temporelle du flux
    (§4.4) — même statistique, cadence différente selon le flux. Catalogue
    minimal pour le POC (une seule famille court terme, cf. Phase 7) ;
    extensible sans restructuration."""
    return CONFIG["cadences_sprt"].get(type_flux, CONFIG["cadences_sprt"]["defaut"])
