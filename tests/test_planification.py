"""Modèle de récompense (g de A*, étape 18) : apprend r̂(champ,action) sur mémoire de
rejeu + mini-lot, et le choix glouton préfère l'action de meilleure récompense prédite."""
import numpy as np

from scl.planification import ModeleRecompense, choisir_glouton


def _champ():
    c = np.zeros((10, 10), dtype=np.float32); c[5, 5] = 0.25
    return c


def test_apprend_a_ordonner_les_actions():
    # récompense synthétique : action 1 bonne (+1), action 0 mauvaise (−1), sur le même champ.
    m = ModeleRecompense(n_actions=2)
    c = _champ()
    for _ in range(300):
        m.observer(c, 1, +1.0)
        m.observer(c, 0, -1.0)
    assert m.predire(c, 1) > m.predire(c, 0)             # a appris l'ordre
    assert choisir_glouton(m, c, 2, epsilon=0.0) == 1     # glouton → la bonne action


def test_predire_et_observer_bornes():
    m = ModeleRecompense(n_actions=5)
    c = _champ()
    perte = m.observer(c, 3, 0.4)
    assert perte >= 0.0
    assert isinstance(m.predire(c, 3), float)
    a = choisir_glouton(m, c, 5, epsilon=1.0)             # ε=1 → action aléatoire valide
    assert 0 <= a < 5
