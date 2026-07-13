"""Phase 6 — tests du pipeline de création : simulateur.py en isolation, puis
porte d'isolement obligatoire du pipeline complet (localisation du point de
branchement → création jumelée) AVANT toute intégration à la boucle réelle
(§25 étape 7 du document d'architecture)."""
import torch

from scl.discriminateur import Discriminateur
from scl.graphe import Graphe
from scl.module import Module
from scl.simulateur import Simulateur


def _module(id_, n_in=4, n_lat=3, **kw):
    return Module(id_, n_inputs_reco=n_in, n_latent=n_lat, n_outputs_gen=n_in, **kw)


# --------------------------------------------------------------- Simulateur seul

def test_entrainement_reduit_la_perte():
    torch.manual_seed(0)
    s = Simulateur("s", dim_contexte_echec=4, dim_latent_stocke=3)
    z, x = torch.randn(3), torch.randn(4)
    pertes = [s.entrainer(z, x) for _ in range(200)]
    # comparaison lissée (début/fin) : la perte gaussienne (terme log σ inclus)
    # n'est pas strictement monotone pas à pas, mais doit baisser nettement.
    debut = sum(pertes[:10]) / 10
    fin = sum(pertes[-10:]) / 10
    assert fin < debut


def test_refabriquer_sans_z_utilise_l_episode_fondateur():
    torch.manual_seed(0)
    s = Simulateur("s", dim_contexte_echec=4, dim_latent_stocke=3)
    z_fondateur, x_echec = torch.randn(3), torch.randn(4)
    s.initialiser_depuis_episode(z_fondateur, x_echec, n_pas=200)
    mu, sigma = s.refabriquer()
    assert mu.shape == (4,)
    assert (sigma > 0).all()
    assert torch.mean((mu - x_echec) ** 2) < 0.5   # ancré sur l'épisode fondateur


def test_refabriquer_sans_z_ni_fondateur_leve_erreur():
    s = Simulateur("s", dim_contexte_echec=4, dim_latent_stocke=3)
    try:
        s.refabriquer()
        assert False, "devait lever ValueError"
    except ValueError:
        pass


def test_generer_contrefactuel_produit_n_variantes_jugees_par_d_phi():
    torch.manual_seed(0)
    s = Simulateur("s", dim_contexte_echec=4, dim_latent_stocke=3)
    d = Discriminateur(dimension=4)
    variantes = s.generer_contrefactuel(torch.randn(3), d, n=5)
    assert len(variantes) == 5
    for v in variantes:
        assert v["chemin"].shape == (4,)
        assert 0.0 < v["plausibilite"] < 1.0
        assert v["provenance"] == "imagine"


def test_est_hors_distribution_dependant_du_seuil():
    s = Simulateur("s", dim_contexte_echec=4, dim_latent_stocke=3)
    d = Discriminateur(dimension=4)
    resultat = s.est_hors_distribution(torch.randn(4), d, seuil=0.0)
    assert resultat is False   # seuil à 0 : rien ne peut être sous le seuil
    resultat = s.est_hors_distribution(torch.randn(4), d, seuil=1.0)
    assert resultat is True    # seuil à 1 : tout est sous le seuil


# ----------------------------------------------- Porte d'isolement du pipeline complet

def test_pipeline_creation_jumelee_isole_de_la_boucle_reelle():
    """Reproduit §4.5 étapes 1→5 en isolation : un graphe synthétique avec un
    point de branchement localisable, puis la création jumelée à cet endroit
    précis. Ce test doit passer avant tout câblage dans boucle.py (Phase 11)."""
    torch.manual_seed(0)
    g = Graphe()
    ctx = torch.zeros(4)
    cap = _module("capteur", n_in=4)
    m1 = _module("m1")
    m2 = _module("m2")   # va s'effondrer : c'est ici que la création doit se produire
    g.ajouter_module(cap, input_node=True)
    g.ajouter_module(m1, parents=["capteur"])
    g.ajouter_module(m2, parents=["m1"])

    for _ in range(6):
        cap._enregistrer_erreur(ctx, erreur=0.0, t=0)
        m1._enregistrer_erreur(ctx, erreur=0.0, t=0)
        m2._enregistrer_erreur(ctx, erreur=1.0, t=0)   # effondré, antécédent sain

    point = g.localiser_point_branchement(ctx)
    assert point == "m2"

    contexte_echec = torch.randn(4)
    resultat = g.creer_module_candidat(point, n_inputs=4, n_latent=3,
                                       contexte_echec=contexte_echec, t=0)
    assert resultat is not None
    module, sim = resultat

    # statut provisoire : jamais de plancher certifié sur la seule base du
    # rejeu simulé (§1.4) — même si le condensateur atteint le seuil de verrou
    for _ in range(200):
        module.mettre_a_jour_condensateurs(erreur_reco=0.0, erreur_gen=0.0)
    assert not module.locked_reco and not module.locked_gen
    assert module.provisoire is True
    assert module.status == "en_test"

    # le simulateur jumeau est bien ancré sur l'épisode fondateur réel
    assert g.simulateurs[module.id] is sim
    assert sim.z_fondateur is not None
    mu, sigma = sim.refabriquer()
    assert mu.shape == (4,)
    assert (sigma > 0).all()
