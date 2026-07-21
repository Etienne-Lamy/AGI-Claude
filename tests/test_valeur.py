"""Valeur Q(champ,action) par TD (étape 19) : le bootstrap remonte une récompense vers
l'action qui y mène ; l'argmax choisit alors l'action de meilleure valeur."""
import random

import numpy as np
import torch

from scl.planification import ModeleValeurQ


def _s(k):
    c = np.zeros((10, 10), dtype=np.float32); c[5, 5] = 0.25; c[0, k] = 1.0
    return c


def test_td_propage_la_recompense_vers_la_bonne_action():
    # depuis s0, action 1 mène à un état s1 récompensé ; action 0 n'apporte rien.
    # seed explicite : le test est stochastique (init réseau + échantillonnage du rejeu).
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    q = ModeleValeurQ(n_actions=2, gamma=0.9)
    s0, s1 = _s(0), _s(1)
    for _ in range(600):
        q.observer(s0, 1, 1.0, s1)      # bonne action : +1
        q.observer(s0, 0, 0.0, s0)      # mauvaise : 0
    qv = q.q(s0)
    assert float(qv[1]) > float(qv[0]) + 0.05      # la valeur privilégie NETTEMENT l'action 1
    assert q.choisir(s0, epsilon=0.0) == 1


def test_choisir_et_observer_bornes():
    q = ModeleValeurQ(n_actions=5)
    s = _s(0)
    perte = q.observer(s, 2, -0.5, _s(2))
    assert perte >= 0.0
    assert 0 <= q.choisir(s, epsilon=1.0) < 5
    assert q.q(s).shape[0] == 5
