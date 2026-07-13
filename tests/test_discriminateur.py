"""Phase 3 — tests du discriminateur partagé D_φ : séparation réel/faux
après entraînement contrastif, atténuation douce jamais nulle."""
import torch

from scl.discriminateur import Discriminateur, attenuer_soft


def test_plausibilite_dans_les_bornes():
    d = Discriminateur(dimension=8)
    p = d.evaluer_plausibilite(torch.randn(8))
    assert 0.0 < p < 1.0


def test_dimension_heterogene_geree_par_projection():
    d = Discriminateur(dimension=8)
    # un vecteur d'une tout autre dimension (ex. latent d'un autre module)
    # doit quand même produire une plausibilité valide, sans erreur.
    p = d.evaluer_plausibilite(torch.randn(37))
    assert 0.0 < p < 1.0


def test_entrainement_contrastif_separe_reel_et_faux():
    torch.manual_seed(0)
    d = Discriminateur(dimension=8)

    def echantillon_reel():
        return torch.randn(8) + 3.0

    def echantillon_faux():
        return torch.randn(8) - 3.0

    for _ in range(300):
        positif = echantillon_reel()
        negatifs = [echantillon_faux() for _ in range(4)]
        d.entrainer_contrastif(positif, negatifs)

    plausibilite_reel = sum(d.evaluer_plausibilite(echantillon_reel()) for _ in range(20)) / 20
    plausibilite_faux = sum(d.evaluer_plausibilite(echantillon_faux()) for _ in range(20)) / 20

    assert plausibilite_reel > plausibilite_faux
    assert plausibilite_reel > 0.7
    assert plausibilite_faux < 0.3


def test_attenuer_soft_jamais_nul():
    for rang in (0, 1, 5, 50, 1000):
        assert attenuer_soft(1.0, rang) > 0.0


def test_attenuer_soft_decroissant_avec_le_rang():
    valeurs = [attenuer_soft(1.0, rang) for rang in range(10)]
    assert all(valeurs[i] > valeurs[i + 1] for i in range(len(valeurs) - 1))


def test_attenuer_soft_rang_zero_inchange():
    assert attenuer_soft(1.0, rang=0) == 1.0
