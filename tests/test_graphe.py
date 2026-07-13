"""Phase 4 — tests de graphe.py : test_non_inferiorite (plancher jamais
plafond), localiser_point_branchement (remplace detecter_rupture),
détachement inter-module, croissance_gouvernee, rejet_gouverne, atrophie,
garde-fous de création, et smoke tests de fragmentation/découpe."""
import torch

from scl.config import CONFIG
from scl.graphe import Graphe, croissance_gouvernee, rejet_gouverne
from scl.graphe import test_non_inferiorite as non_inferiorite
from scl.memoires import MemoireTampon, RegistreProvenance, RegistreRupture
from scl.module import Module


def _module(id_, n_in=4, n_lat=3, **kw):
    return Module(id_, n_inputs_reco=n_in, n_latent=n_lat, n_outputs_gen=n_in, **kw)


# ---------------------------------------------------------- test_non_inferiorite

def test_non_inferiorite_accepte_candidat_clairement_meilleur():
    a = [0.5] * 20
    b = [0.1] * 20
    assert non_inferiorite(a, b) is True


def test_non_inferiorite_rejette_candidat_clairement_pire():
    a = [0.1] * 20
    b = [0.9] * 20
    assert non_inferiorite(a, b) is False


def test_non_inferiorite_accepte_indistinguable_plancher_pas_plafond():
    # plancher jamais plafond (§1.4/§0) : une différence non significative
    # (bruit) doit être ACCEPTÉE, pas bloquée par principe.
    torch.manual_seed(0)
    a = [0.20 + 0.01 * torch.randn(1).item() for _ in range(30)]
    b = [0.20 + 0.01 * torch.randn(1).item() for _ in range(30)]
    assert non_inferiorite(a, b) is True


def test_non_inferiorite_flux_vide_refuse():
    assert non_inferiorite([], []) is False


# -------------------------------------------------------- localiser_point_branchement

def _seeder_erreur(module, contexte, erreur, n=6):
    for _ in range(n):
        module._enregistrer_erreur(contexte, erreur, t=0)


def test_localiser_point_branchement_identifie_le_premier_point_effondre():
    g = Graphe()
    ctx = torch.zeros(4)
    cap = _module("capteur", n_in=4)
    m1 = _module("m1")
    m2 = _module("m2")
    m3 = _module("m3")
    g.ajouter_module(cap, input_node=True)
    g.ajouter_module(m1, parents=["capteur"])
    g.ajouter_module(m2, parents=["m1"])
    g.ajouter_module(m3, parents=["m2"])

    _seeder_erreur(cap, ctx, erreur=0.0)   # sain
    _seeder_erreur(m1, ctx, erreur=0.0)    # sain
    _seeder_erreur(m2, ctx, erreur=1.0)    # effondré, antécédent (m1) sain
    _seeder_erreur(m3, ctx, erreur=1.0)    # effondré aussi, mais pas le premier

    assert g.localiser_point_branchement(ctx) == "m2"


def test_localiser_point_branchement_capteur_en_tete():
    g = Graphe()
    ctx = torch.zeros(4)
    cap = _module("capteur", n_in=4)
    m1 = _module("m1")
    g.ajouter_module(cap, input_node=True)
    g.ajouter_module(m1, parents=["capteur"])
    _seeder_erreur(cap, ctx, erreur=1.0)   # le capteur lui-même est effondré
    _seeder_erreur(m1, ctx, erreur=1.0)
    assert g.localiser_point_branchement(ctx) == "capteur:capteur"


def test_localiser_point_branchement_rien_si_tout_sain():
    g = Graphe()
    ctx = torch.zeros(4)
    cap = _module("capteur", n_in=4)
    g.ajouter_module(cap, input_node=True)
    _seeder_erreur(cap, ctx, erreur=0.0)
    assert g.localiser_point_branchement(ctx) is None


def test_localiser_point_branchement_ignore_reflexe_cable_verrouille():
    g = Graphe()
    ctx = torch.zeros(4)
    reflexe = _module("reflexe", innate=True)
    reflexe.locked_reco = True
    reflexe.locked_gen = True
    g.ajouter_module(reflexe, input_node=True)
    _seeder_erreur(reflexe, ctx, erreur=1.0)   # "effondré" mais câblé : ignoré
    assert g.localiser_point_branchement(ctx) is None


# ---------------------------------------------------------------- entree_detachee

def test_entree_detachee_ne_transmet_aucun_gradient():
    g = Graphe()
    en_amont = torch.randn(4, requires_grad=True)
    detachee = g.entree_detachee([en_amont])
    assert not detachee.requires_grad
    assert detachee.grad_fn is None


# ------------------------------------------------------------- croissance_gouvernee

def test_croissance_gouvernee_accepte_baisse_stricte():
    assert croissance_gouvernee(phi_avant=1.0, phi_apres=0.5) is True


def test_croissance_gouvernee_rejette_baisse_negligeable():
    assert croissance_gouvernee(phi_avant=1.0, phi_apres=0.9999) is False


def test_croissance_gouvernee_rejette_hausse():
    assert croissance_gouvernee(phi_avant=1.0, phi_apres=1.5) is False


# ----------------------------------------------------------------- rejet_gouverne

def test_rejet_gouverne_conserve_si_gate_informatif():
    # H1 (gate riche) explique BEAUCOUP mieux les données que H0 (non informatif)
    assert rejet_gouverne(log_vraisemblance_h0=-100.0, log_vraisemblance_h1=-50.0,
                          df=1) == "conserver"


def test_rejet_gouverne_rejette_si_gate_non_informatif():
    assert rejet_gouverne(log_vraisemblance_h0=-50.0, log_vraisemblance_h1=-50.0,
                          df=1) == "rejeter"


# ------------------------------------------------------------------------ atrophie

def test_atrophier_ignore_module_inne():
    g = Graphe()
    m = _module("m", innate=True)
    m.tentatives_count = 10_000
    m.condensateur_reco = m.condensateur_gen = 0.0
    assert g.atrophier(m) is False


def test_atrophier_ignore_module_immature():
    g = Graphe()
    m = _module("m")
    m.tentatives_count = 1
    m.condensateur_reco = m.condensateur_gen = 0.0
    assert g.atrophier(m) is False


def test_atrophier_purge_la_provenance_d_un_module_provisoire():
    g = Graphe()
    m = _module("m", provisoire=True)
    m.tentatives_count = CONFIG["maturite_structurelle"] + 1
    m.condensateur_reco = m.condensateur_gen = 0.0
    rp = RegistreProvenance()
    rp.marquer({"module_id": "m", "valeur": 1}, "imagine")
    rp.marquer({"module_id": "m", "valeur": 2}, "imagine")
    assert g.atrophier(m, registre_provenance=rp) is True
    assert m.status == "abandonné"
    assert rp.purger("m") == 0   # déjà purgé par atrophier


def test_atrophier_ne_declenche_pas_si_condensateurs_corrects():
    g = Graphe()
    m = _module("m")
    m.tentatives_count = CONFIG["maturite_structurelle"] + 1
    m.condensateur_reco = m.condensateur_gen = 0.8
    assert g.atrophier(m) is False


# ------------------------------------------------------------ creer_module_candidat

def test_creer_module_candidat_un_seul_a_la_fois():
    g = Graphe()
    m = _module("point")
    g.ajouter_module(m)
    c1 = g.creer_module_candidat("point", n_inputs=4, n_latent=3, t=0)
    assert c1 is not None
    module, sim = c1
    assert module.status == "en_test"
    assert module.provisoire is True
    assert sim.id.startswith("point_candidat_0")
    c2 = g.creer_module_candidat("point", n_inputs=4, n_latent=3, t=1)
    assert c2 is None   # un candidat est déjà en_test


def test_creer_module_candidat_respecte_le_cooldown():
    g = Graphe()
    m = _module("point")
    g.ajouter_module(m)
    rr = RegistreRupture()
    rr.marquer_abandon("point", t=0)
    c = g.creer_module_candidat("point", n_inputs=4, n_latent=3,
                                registre_rupture=rr, t=1)
    assert c is None


# ---------------------------------------------------- smoke : fragmentation / découpe

def test_fragmenter_module_smoke():
    from scl.memoires import RegistreCablage
    g = Graphe()
    m = _module("m")
    g.ajouter_module(m)
    rc = RegistreCablage()
    regle, exception = g.fragmenter_module(m, rc, contexte_t=None, t=0)
    assert regle.id == "m"
    assert exception.status == "en_test"
    assert "m" in g.modules and exception.id in g.modules


def test_controle_multiplicite_delegue_a_statistiques():
    g = Graphe()
    tests_du_jour = [{"p_valeur": 0.001}, {"p_valeur": 0.002}, {"p_valeur": 0.9}]
    indices, seuil = g.controle_multiplicite(tests_du_jour)
    assert set(indices) == {0, 1}


def test_recalage_plancher_drift_deverrouille_sur_h1():
    g = Graphe()
    m = _module("m")
    m.locked_reco = m.locked_gen = True
    m.plancher_reco = m.plancher_gen = 0.95
    assert g.recalage_plancher_drift(m, ("H1", 5)) is True
    assert not m.locked_reco and not m.locked_gen
    assert m.plancher_reco is None and m.plancher_gen is None


def test_recalage_plancher_drift_ne_touche_rien_sinon():
    g = Graphe()
    m = _module("m")
    m.locked_reco = m.locked_gen = True
    m.plancher_reco = m.plancher_gen = 0.95
    assert g.recalage_plancher_drift(m, ("continuer", 3)) is False
    assert m.locked_reco and m.locked_gen


def test_decoupe_et_validation_smoke():
    torch.manual_seed(0)
    g = Graphe()
    m = _module("m", n_in=4, n_lat=3)
    g.ajouter_module(m)

    ctx_a = torch.tensor([5.0, 0.0, 0.0, 0.0])
    ctx_b = torch.tensor([-5.0, 0.0, 0.0, 0.0])
    for _ in range(15):
        m._enregistrer_erreur(ctx_a + torch.randn(4) * 0.01, erreur=0.01, t=0)
        m._enregistrer_erreur(ctx_b + torch.randn(4) * 0.01, erreur=0.5, t=0)

    mt = MemoireTampon()
    resultat = g.decouper_module(m, mt, t=0)
    assert resultat is not None
    noyau, amovible = resultat
    assert noyau.id == "m"
    assert amovible.status == "en_test"

    for _ in range(10):
        mt.ajouter_reco("m", ctx_a + torch.randn(4) * 0.01, torch.randn(3), erreur=0.01, t=0)
        mt.ajouter_reco("m", ctx_b + torch.randn(4) * 0.01, torch.randn(3), erreur=0.5, t=0)

    decision = g.valider_decoupe(noyau, amovible, mt)
    assert decision in ("intégrer", "abandonner_amovible", "fusionner_retour")
