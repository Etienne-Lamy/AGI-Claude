"""Planification A* sur l'état-objet (étape 25) : trouver la séquence d'accélérations qui
amène un sucre sur le corps."""
import numpy as np

from scl.dynamique_objet import DynamiqueObjet
from scl.monde import VAL_CORPS, VAL_BATON, VAL_SUCRE
from scl.perception_objet import ChampObjet
from scl.planification_objet import chercher_sucre, _avancer


def _po():
    po = ChampObjet()
    po.val_cat = np.array([0.0, VAL_CORPS, VAL_BATON, VAL_SUCRE], dtype=np.float32)
    po.cat_corps, po.cat_objet = 1, {2, 3}          # 2=bâton 3=sucre
    return po


def test_a_etoile_trouve_un_plan_vers_le_sucre():
    po = _po(); dyn = DynamiqueObjet(po)
    objets = [(3, (7, 5))]                            # sucre à l'offset (2,0)
    plan = chercher_sucre(po, dyn, objets, v=(0, 0))
    assert plan is not None and len(plan) <= 4
    # exécuter le plan sur le modèle : le sucre doit être mangé
    E, v, mange = objets, (0, 0), False
    for a in plan:
        v = dyn.accel(v, a)
        m, _, E = _avancer(po, E, v, {3}, {2})
        mange = mange or m
    assert mange


def test_pas_de_sucre_pas_de_plan():
    po = _po(); dyn = DynamiqueObjet(po)
    assert chercher_sucre(po, dyn, [(2, (7, 5))], v=(0, 0)) is None   # que du bâton
