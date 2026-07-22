"""Perception objet (étape 23) : la géométrie décalage/régénération (prédiction = décaler
les objets par la vitesse). Le VQ (catégorisation émergente) est validé par etape23."""
import numpy as np

from scl.perception_objet import ChampObjet
from scl.monde import VAL_CORPS, VAL_SUCRE


def _po():
    po = ChampObjet()
    po.val_cat = np.array([0.0, VAL_CORPS, 0.5, VAL_SUCRE], dtype=np.float32)
    po.cat_corps, po.cat_objet = 1, {2, 3}       # 0=vide 1=corps 2=bâton 3=sucre
    return po


def test_regenere_corps_au_centre_et_objets():
    po = _po()
    f = po.regenerer([(3, (3, 4))])              # un sucre en (3,4)
    assert f[po.centre, po.centre] == VAL_CORPS   # corps toujours au centre
    assert f[3, 4] == VAL_SUCRE
    assert f.sum() == VAL_CORPS + VAL_SUCRE       # rien d'autre


def test_decale_par_la_vitesse():
    po = _po()
    d = po.decaler([(3, (3, 4))], (1, 0))         # v=(1,0) → position i-1
    assert d == [(3, (2, 4))]
    # objet qui sort du cadre → disparaît (non prévisible)
    assert po.decaler([(3, (0, 4))], (1, 0)) == []


def test_predire_est_decalage_regenere():
    po = _po()
    pred = po.regenerer(po.decaler([(3, (4, 4))], (1, 0)))   # sucre (4,4) décalé de (1,0)
    assert pred[3, 4] == VAL_SUCRE                            # → (3,4)
    assert pred[4, 4] == 0.0
