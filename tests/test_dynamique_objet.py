"""Dynamique objet (étape 24) : action = accélération (clip), et compositionnalité
translater(·,(2,0)) = translater(·,(1,0)) appliqué deux fois."""
from scl.dynamique_objet import DynamiqueObjet
from scl.perception_objet import ChampObjet


def test_accel_est_un_changement_de_vitesse_borne():
    dyn = DynamiqueObjet(None, v_max=2)
    assert dyn.accel((1, 0), (1, 0)) == (2, 0)        # accélérer
    assert dyn.accel((2, 0), (1, 0)) == (2, 0)        # borné à v_max
    assert dyn.accel((0, 0), (-1, 0)) == (-1, 0)
    assert dyn.accel((-2, 0), (-1, 0)) == (-2, 0)     # borné en négatif


def test_compositionnalite_2_0_est_1_0_deux_fois():
    po = ChampObjet()
    dyn = DynamiqueObjet(po)
    objs = [(3, (6, 4)), (2, (5, 7))]                  # deux objets (hors centre)
    direct = po.decaler(objs, (2, 0))                  # module (2,0) direct
    compose = dyn.translater_compose(objs, (2, 0))     # (1,0) appliqué 2×
    assert sorted(direct) == sorted(compose)           # IDENTIQUE : pas de module (2,0) séparé


def test_transition_accelere_puis_translate():
    po = ChampObjet()
    dyn = DynamiqueObjet(po)
    E, v = dyn.transition([(3, (6, 4))], (0, 0), (1, 0))   # v:0→1, décale de 1
    assert v == (1, 0) and E == [(3, (5, 4))]
