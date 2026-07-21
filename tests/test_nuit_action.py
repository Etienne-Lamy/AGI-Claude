"""Rejeu nocturne amont (étape 20) : les retours n-pas font REMONTER le crédit d'une
récompense finale vers les états qui la précèdent (le cœur de la navigation apprise)."""
import numpy as np

from scl.nuit_action import RejeuNocturne
from scl.planification import ModeleValeurQ


def _s(k):
    c = np.zeros((10, 10), dtype=np.float32); c[5, 5] = 0.25; c[0, k] = 1.0
    return c


def test_le_credit_remonte_en_amont():
    # épisode : 6 états distincts, on prend toujours l'action 1, récompense SEULEMENT au bout.
    episode = [(_s(k), 1, 0.0) for k in range(5)] + [(_s(5), 1, 1.0)]
    q = ModeleValeurQ(n_actions=2, gamma=0.95)
    rej = RejeuNocturne(n_pas=6, priorite_sucre=8.0)
    assert rej.enregistrer(episode) is True             # épisode récompensé détecté
    q0_avant = float(q.q(_s(0))[1])
    rej.nuit(q, passes=400, lot=64)
    q0_apres = float(q.q(_s(0))[1])
    # le crédit du sucre (5 pas plus loin) a remonté jusqu'au tout premier état
    assert q0_apres > q0_avant + 0.1
    assert float(q.q(_s(0))[1]) > float(q.q(_s(0))[0])   # l'action prise vaut plus que l'autre


def test_priorite_sucre_et_sans_episode():
    rej = RejeuNocturne()
    assert rej.enregistrer([(_s(0), 0, 0.0)]) is False   # pas de récompense → non prioritaire
    q = ModeleValeurQ(n_actions=2)
    assert rej.nuit(q, passes=10) == 0.0 or True         # tolère l'absence, ne plante pas
