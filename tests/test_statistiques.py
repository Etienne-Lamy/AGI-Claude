"""Phase 5 — tests de statistiques.py : résidu normalisé, les trois usages
du SPRT (surprise/création/drift, à vérité connue), contrôle FDR, cadence
variable."""
import torch

from scl.module import Module
from scl.statistiques import (
    residu_normalise, residu_module, sprt_surprise, sprt_creation,
    sprt_drift, controle_fdr, cadence_variable,
)


# ------------------------------------------------------------- résidu normalisé

def test_residu_normalise_nul_si_x_egal_mu():
    assert residu_normalise(torch.zeros(4), torch.zeros(4), sigma=1.0) == 0.0


def test_residu_normalise_formule_isotrope():
    x = torch.tensor([2.0, 0.0])
    mu = torch.tensor([0.0, 0.0])
    # ||x-mu||^2 = 4, sigma^2 = 2 -> résidu = 4/2 = 2
    assert abs(residu_normalise(x, mu, sigma=2.0 ** 0.5) - 2.0) < 1e-6


def test_residu_module_smoke():
    m = Module("m", n_inputs_reco=4, n_latent=3, n_outputs_gen=4)
    r = residu_module(m, torch.randn(4), torch.randn(4))
    assert r >= 0.0


# ---------------------------------------------------------------- sprt_surprise

def test_sprt_surprise_conclut_h1_sur_residus_systematiquement_eleves():
    d = 10
    mu1 = float(d) + 5.0   # au niveau exact de H1 (decalage par défaut = 0.5*d)
    residus = [mu1] * 10
    decision, n = sprt_surprise(residus, d=d)
    assert decision == "H1"
    assert n <= 10


def test_sprt_surprise_conclut_h0_sur_residus_normaux():
    d = 10
    residus = [float(d)] * 10   # exactement la moyenne attendue sous H0
    decision, n = sprt_surprise(residus, d=d)
    assert decision == "H0"


# ---------------------------------------------------------------- sprt_creation

def test_sprt_creation_declenche_sur_contextes_distincts():
    d = 10
    residu_eleve = float(d) + 5.0
    echecs = [(torch.ones(4) * i * 10, residu_eleve) for i in range(10)]   # contextes distincts
    decision, n = sprt_creation(echecs, d=d)
    assert decision == "H1"


def test_sprt_creation_ne_declenche_pas_sur_contexte_repete():
    d = 10
    residu_eleve = float(d) + 5.0
    meme_contexte = torch.ones(4)
    echecs = [(meme_contexte, residu_eleve) for _ in range(10)]   # même contexte, 10 fois
    decision, n = sprt_creation(echecs, d=d)
    assert decision != "H1"   # un seul contexte distinct : pas assez de preuve


# ------------------------------------------------------------------- sprt_drift

def test_sprt_drift_detecte_translation():
    anciens = [1.0] * 20
    recents = [1.0 + 10.0] * 10   # translation nette et soutenue
    decision, n = sprt_drift(anciens, recents)
    assert decision == "H1"


def test_sprt_drift_rien_si_meme_distribution():
    anciens = [1.0, 1.1, 0.9, 1.05, 0.95] * 4
    recents = [1.0, 1.1, 0.9, 1.05, 0.95] * 4
    decision, n = sprt_drift(anciens, recents)
    assert decision != "H1"


def test_sprt_drift_sans_historique_continue():
    assert sprt_drift([], [1.0, 2.0]) == ("continuer", 0)


# --------------------------------------------------------------------- controle_fdr

def test_controle_fdr_exemple_connu():
    p_valeurs = [0.001, 0.002, 0.20, 0.50, 0.80]
    indices, seuil = controle_fdr(p_valeurs, alpha=0.05)
    assert len(indices) == 2
    assert set(indices) == {0, 1}   # les deux p-valeurs les plus basses


def test_controle_fdr_liste_vide():
    assert controle_fdr([]) == ([], 0.0)


def test_controle_fdr_rien_de_significatif():
    p_valeurs = [0.9, 0.8, 0.95]
    indices, seuil = controle_fdr(p_valeurs, alpha=0.05)
    assert indices == []


# ----------------------------------------------------------------- cadence_variable

def test_cadence_variable_connue_et_defaut():
    assert cadence_variable("sensorimoteur") == 1
    assert cadence_variable("inconnu_xyz") == cadence_variable("defaut")
