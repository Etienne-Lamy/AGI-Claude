"""Phase 7 — tests de memoire_travail.py : TamponRelatif (indexation
purement relative — test structurel explicite), cycle émission→maturation de
FamilleHorizon, PalierSommeil gated par D_φ."""
import pytest
import torch

from scl.discriminateur import Discriminateur
from scl.memoire_travail import (
    TamponRelatif, hierarchie_deux_vitesses, fenetre_glissante_continue,
    FamilleHorizon, PalierSommeil,
)
from scl.module import Module
from scl.simulateur import Simulateur


# ------------------------------------------------------------------ TamponRelatif

def test_lecture_ecriture_bornes():
    t = TamponRelatif(K=2)
    t.ecrire(0, "a")
    t.ecrire(2, "b")
    t.ecrire(-2, "c")
    assert t.lire(0) == "a" and t.lire(2) == "b" and t.lire(-2) == "c"
    with pytest.raises(IndexError):
        t.lire(3)
    with pytest.raises(IndexError):
        t.ecrire(-3, "x")


def test_decaler_deplace_le_reel_vers_le_passe_au_fil_des_pas():
    t = TamponRelatif(K=2)
    t.decaler("t0")
    t.decaler("t1")
    assert t.lire(0) == "t1"
    assert t.lire(-1) == "t0"


def test_decaler_retourne_la_prediction_arrivee_a_maturite_avant_ecrasement():
    # une prédiction n'est jamais écrite à δ=0 (réservé au réel via decaler) :
    # elle est écrite à δ=+1 (émise "pour le pas suivant"), et c'est SON
    # arrivée à δ=0, un decaler plus tard, que la fonction doit renvoyer.
    t = TamponRelatif(K=2)
    t.ecrire(1, "prediction")
    arrivee = t.decaler("nouveau_reel")
    assert arrivee == "prediction"    # renvoyée AVANT d'être écrasée
    assert t.lire(0) == "nouveau_reel"   # δ=0 porte maintenant le RÉEL


def test_decaler_rien_a_maturite_si_rien_n_etait_predit():
    t = TamponRelatif(K=2)
    assert t.decaler("reel") is None


def test_decaler_perd_le_contenu_sortant_a_moins_k():
    t = TamponRelatif(K=1)
    t.ecrire(-1, "sur_le_point_de_sortir")
    t.decaler("r1")
    # après un décalage, l'ancien δ=-1 est perdu, aucune trace n'en reste
    assert t.lire(-1) != "sur_le_point_de_sortir"


def test_structure_purement_relative_aucun_index_absolu():
    """Contrat structurel du §0 : TamponRelatif ne porte AUCUN compteur de
    temps absolu — sa taille et sa sémantique d'offset restent identiques
    après un nombre arbitraire de décalages."""
    t = TamponRelatif(K=3)
    attrs_temporels = {"t", "step", "temps", "temps_absolu", "horloge", "compteur"}
    assert not (attrs_temporels & set(vars(t).keys()))
    for i in range(5000):
        t.decaler(i)   # la VALEUR stockée peut être un compteur (payload de
                       # l'appelant), mais l'INDEX, lui, ne l'est jamais
    assert t.taille == 7
    assert len(t._contenu) == 7
    with pytest.raises(IndexError):
        t.lire(4)
    with pytest.raises(IndexError):
        t.lire(-4)
    assert t.lire(0) == 4999    # dernière valeur écrite : toujours à δ=0
    assert t.lire(-1) == 4998
    assert t.lire(-3) == 4996   # borne basse : le plus ancien encore conservé


def test_hierarchie_deux_vitesses():
    rapide, lent = hierarchie_deux_vitesses(K_rapide=4, K_lent=10)
    assert rapide.K == 4
    assert lent.K == 10


def test_fenetre_glissante_continue_est_un_alias_de_decaler():
    t1, t2 = TamponRelatif(K=2), TamponRelatif(K=2)
    t1.ecrire(1, "x")
    t2.ecrire(1, "x")
    a1 = t1.decaler("r")
    a2 = fenetre_glissante_continue(t2, "r")
    assert a1 == a2
    assert t1.lire(0) == t2.lire(0)


# -------------------------------------------------------------------- FamilleHorizon

def test_famille_horizon_cycle_emission_maturation():
    torch.manual_seed(0)
    m = Module("m", n_inputs_reco=4, n_latent=3, n_outputs_gen=4)
    fh = FamilleHorizon("court_terme", h=3)
    entree = torch.randn(4)
    fh.emettre(m, entree, t=0)

    residu1 = fh.maturer(torch.randn(4), t=1)
    residu2 = fh.maturer(torch.randn(4), t=2)
    assert residu1 is None and residu2 is None   # pas encore à échéance

    reel_final = torch.randn(4)
    residu3 = fh.maturer(reel_final, t=3)
    assert residu3 is not None and residu3 >= 0.0   # échéance atteinte (δ=+3 -> 0)


def test_famille_horizon_respecte_la_cadence():
    torch.manual_seed(0)
    m = Module("m", n_inputs_reco=4, n_latent=3, n_outputs_gen=4)
    fh = FamilleHorizon("lente", h=2, cadence=3)
    entree = torch.randn(4)
    assert fh.emettre(m, entree, t=0) is None    # 1er pas : pas encore la cadence
    assert fh.emettre(m, entree, t=1) is None    # 2e pas : toujours pas
    assert fh.emettre(m, entree, t=2) is not None  # 3e pas : cadence atteinte


# ---------------------------------------------------------------------- PalierSommeil

def test_palier_sommeil_recupere_si_plausible():
    torch.manual_seed(0)
    sim = Simulateur("s", dim_contexte_echec=4, dim_latent_stocke=3)
    d = Discriminateur(dimension=4)
    d.evaluer_plausibilite = lambda x: 0.9   # verdict D_φ forcé : plausible

    ps = PalierSommeil()
    ps.stocker(torch.randn(3), t=0)
    resultat = ps.recuperer(sim, d, seuil_plausibilite=0.5)
    assert resultat is not None
    assert resultat.shape == (4,)
    assert ps._stockes[0]["resolu"] is True


def test_palier_sommeil_refuse_si_implausible():
    torch.manual_seed(0)
    sim = Simulateur("s", dim_contexte_echec=4, dim_latent_stocke=3)
    d = Discriminateur(dimension=4)
    d.evaluer_plausibilite = lambda x: 0.1   # verdict D_φ forcé : implausible

    ps = PalierSommeil()
    ps.stocker(torch.randn(3), t=0)
    resultat = ps.recuperer(sim, d, seuil_plausibilite=0.5)
    assert resultat is None
    assert ps._stockes[0]["resolu"] is False
