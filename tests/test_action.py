"""Transition action-conditionnée (étape 16) : un module champ→champ par action,
entraîné sur la transition observée sous l'action émise (copie d'efférence)."""
import numpy as np

from scl.action import TransitionActionChamp


def _champ(shift):
    c = np.zeros((10, 10), dtype=np.float32)
    c[5, 5] = 0.25                      # corps au centre
    c[3, (4 + shift) % 10] = 1.0        # un objet dont la position dépend du décalage
    return c


def test_observe_predit_et_rappel_borne():
    tac = TransitionActionChamp([(0, 0), (1, 0)])
    cp, c = _champ(0), _champ(1)
    tac.observer(cp, (1, 0), c)
    pred = tac.predire(cp, (1, 0))
    assert np.asarray(pred).reshape(-1).shape[0] == 100     # champ 10×10 prédit
    r = tac.rappel(cp, (1, 0), c)
    assert 0.0 <= r <= 1.0
    assert tac.n_maj[(1, 0)] == 1 and tac.n_maj[(0, 0)] == 0


def test_matrice_rappel_structure():
    actions = [(0, 0), (1, 0)]
    tac = TransitionActionChamp(actions)
    trans = [(_champ(0), (0, 0), _champ(0)), (_champ(0), (1, 0), _champ(1))]
    for cp, a, c in trans:
        tac.observer(cp, a, c)
    mat = tac.matrice_rappel(trans)
    assert set(mat) == set(actions)
    for a in actions:
        assert set(mat[a]) == set(actions)
        for b in actions:
            assert 0.0 <= mat[a][b] <= 1.0
