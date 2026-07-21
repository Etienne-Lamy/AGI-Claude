"""Placement maternel (étape 22) : poser l'agent près d'un sucre existant (sans fabriquer
de sucre) pour une récompense dense."""
import numpy as np

from scl.maternage import placer_pres_sucre
from scl.monde import Monde


def test_place_bien_pres_dun_sucre():
    monde = Monde(graine=1)
    rng = np.random.default_rng(0)
    d = placer_pres_sucre(monde, rng, dmin=2, dmax=3)
    assert d is not None                                  # le monde a des sucres
    assert 2 <= max(abs(d[0]), abs(d[1])) <= 3            # à la bonne distance
    sucres, _ = monde.objets_visibles()
    assert d in sucres                                    # le sucre est réellement là (pas fabriqué)
    assert int(monde.vitesse[0]) == 0 and int(monde.vitesse[1]) == 0   # posé à l'arrêt
