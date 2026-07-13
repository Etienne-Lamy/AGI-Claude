"""Phase 2 — tests de module.py : π(x) contextuelle, accumulateur ḡ jour/nuit,
canal réinjecté, statut provisoire, non-franchissement de gradient,
copier_module, évaluations."""
import torch

from scl.config import CONFIG
from scl.module import Module, copier_module


def _petit_module(**kw):
    return Module("t", n_inputs_reco=6, n_latent=3, n_outputs_gen=6, **kw)


# --------------------------------------------------------- fiabilite_contextuelle

def test_fiabilite_contextuelle_sans_historique_est_neutre():
    m = _petit_module()
    assert m.fiabilite_contextuelle(torch.randn(4)) == 0.5


def test_fiabilite_contextuelle_distingue_deux_contextes():
    m = _petit_module()
    ctx_fiable = torch.zeros(4)
    ctx_non_fiable = torch.ones(4) * 10
    for _ in range(8):
        m._enregistrer_erreur(ctx_fiable, erreur=0.0, t=0)          # succès systématique
        m._enregistrer_erreur(ctx_non_fiable, erreur=5.0, t=0)      # échec systématique
    pi_fiable = m.fiabilite_contextuelle(ctx_fiable)
    pi_non_fiable = m.fiabilite_contextuelle(ctx_non_fiable)
    assert pi_fiable > pi_non_fiable
    assert pi_fiable > 0.9
    assert pi_non_fiable < 0.1


# ------------------------------------------------------ accumulateur ḡ (§1.3)

def test_incorporer_gradient_ema_jour_puis_nuit():
    m = _petit_module()
    p0 = m.parametres_reco()[0]
    n = len(m.parametres_reco())

    g1 = torch.ones_like(p0) * 2.0
    m.incorporer_gradient([g1] + [None] * (n - 1), "reco", phase="jour")
    beta_j = CONFIG["beta_jour"]
    attendu = beta_j * 0.0 + (1 - beta_j) * 2.0
    assert torch.allclose(m._g_reco[0], torch.full_like(p0, attendu))

    g2 = torch.ones_like(p0) * 4.0
    m.incorporer_gradient([g2] + [None] * (n - 1), "reco", phase="nuit")
    beta_n = CONFIG["beta_nuit"]
    attendu2 = beta_n * attendu + (1 - beta_n) * 4.0
    assert torch.allclose(m._g_reco[0], torch.full_like(p0, attendu2))


def test_beta_jour_et_nuit_distincts():
    assert CONFIG["beta_jour"] != CONFIG["beta_nuit"]


# --------------------------------------------------------------- canal réinjecté

def test_sortie_generation_inclut_le_canal_reinjecte():
    m = _petit_module()
    sortie = m.forward_generation(torch.randn(m.n_inputs_gen))
    assert sortie.shape == (m.n_outputs_gen + m.dim_reinjection,)
    assert m.dernier_reinjecte.shape == (m.dim_reinjection,)


def test_aligner_action_retourne_la_taille_utile_seulement():
    m = _petit_module()
    commande = m.aligner_action(torch.randn(m.n_outputs_gen), n_iterations=2)
    assert commande.shape == (m.n_outputs_gen,)


# ------------------------------------------------------------------ statut provisoire

def test_module_provisoire_ne_se_verrouille_jamais():
    m = _petit_module(provisoire=True)
    for _ in range(200):
        m.mettre_a_jour_condensateurs(erreur_reco=0.0, erreur_gen=0.0)
    assert m.condensateur_reco >= CONFIG["seuil_verrou"]   # le condensateur monte...
    assert not m.locked_reco                                # ...mais pas de verrou
    assert not m.locked_gen
    assert m.plancher_reco is None


def test_confirmer_reel_leve_le_garde_fou():
    m = _petit_module(provisoire=True)
    for _ in range(200):
        m.mettre_a_jour_condensateurs(erreur_reco=0.0, erreur_gen=0.0)
    assert not m.locked_reco
    m.confirmer_reel()
    assert not m.provisoire
    m.mettre_a_jour_condensateurs(erreur_reco=0.0, erreur_gen=0.0)
    assert m.locked_reco
    assert m.plancher_reco is not None


def test_module_non_provisoire_se_verrouille_normalement():
    m = _petit_module()
    for _ in range(200):
        m.mettre_a_jour_condensateurs(erreur_reco=0.0, erreur_gen=0.0)
    assert m.locked_reco and m.locked_gen
    assert m.plancher_reco == m.condensateur_reco


# ------------------------------------------------- non-franchissement de gradient

def test_aucun_gradient_ne_traverse_la_frontiere_du_module():
    m = _petit_module()
    en_amont = torch.randn(6, requires_grad=True)
    m.entrainer_module_reco(en_amont, torch.randn(3))
    assert en_amont.grad is None   # le module détache ses entrées à l'appel


def test_aligner_action_ne_modifie_aucun_poids_du_decodeur():
    m = _petit_module()
    avant = [p.detach().clone() for p in m.parametres_gen()]
    m.aligner_action(torch.randn(m.n_outputs_gen), n_iterations=5)
    apres = m.parametres_gen()
    assert all(torch.equal(a, p.detach()) for a, p in zip(avant, apres))


# --------------------------------------------------------------------- copier_module

def test_copier_module_detache_les_poids():
    m = _petit_module()
    m.condensateur_reco = 0.42
    copie = copier_module(m, "t2")
    assert copie.id == "t2"
    assert torch.equal(copie.parametres_reco()[0].detach(), m.parametres_reco()[0].detach())
    with torch.no_grad():
        m.parametres_reco()[0].add_(1.0)
    assert not torch.equal(copie.parametres_reco()[0].detach(), m.parametres_reco()[0].detach())
    assert copie.condensateur_reco == 0.42


# --------------------------------------------------------------------- évaluations

def test_evaluer_reco_gen_avec_cle_cible():
    m = _petit_module()
    tentatives_reco = [{"input": torch.randn(6), "cible": torch.randn(3)}]
    tentatives_gen = [{"input": torch.randn(m.n_inputs_gen), "cible": torch.randn(6)}]
    assert m.evaluer_reco(tentatives_reco) >= 0.0
    assert m.evaluer_gen(tentatives_gen) >= 0.0


def test_entrainement_reduit_l_erreur_sur_mapping_fixe():
    torch.manual_seed(0)
    m = _petit_module()
    x, y = torch.randn(6), torch.randn(3)
    erreurs = [m.entrainer_module_reco(x, y) for _ in range(300)]
    assert erreurs[-1] < erreurs[0]
