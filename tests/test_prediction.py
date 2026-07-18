"""ÉTAPE 2a — test léger de la prédiction de transition (module_ae.entrainer_
transition) : un prédicteur entraîné sur un DÉCALAGE fixe prédit ce décalage
mieux qu'un décalage différent (fiabilité = indicateur de vitesse). Frames
synthétiques (rapide, déterministe)."""
import numpy as np

from scl.module_ae import ModuleAutoencodeur


def _champ(rng, t=10):
    c = np.zeros((t, t), dtype=np.float32)
    c[t // 2, t // 2] = 0.25
    for _ in range(3):
        c[rng.integers(t), rng.integers(t)] = 1.0
    for _ in range(2):
        c[rng.integers(t), rng.integers(t)] = 0.5
    return c


def _decale(c, dv):
    return np.roll(np.roll(c, dv[0], axis=0), dv[1], axis=1)


def test_predicteur_apprend_un_decalage_et_le_prefere():
    rng = np.random.default_rng(0)
    pred = ModuleAutoencodeur("test_pred")
    dv = (1, 1)                                   # "vitesse" d'entraînement
    for _ in range(700):
        c = _champ(rng)
        pred.entrainer_transition(c, _decale(c, dv))
    # rappel de prédiction : sur le décalage entraîné vs un autre
    def rappel(dv_test, n=25):
        r = []
        for _ in range(n):
            c = _champ(rng)
            r.append(pred.fidelite_transition(c, _decale(c, dv_test))["rappel"])
        return sum(r) / len(r)
    r_entraine = rappel(dv)
    r_autre = rappel((2, 0))
    assert r_entraine > 0.6, f"n'a pas appris le décalage entraîné : {r_entraine}"
    assert r_entraine > r_autre + 0.2, \
        f"ne discrimine pas la vitesse : entraîné={r_entraine} autre={r_autre}"
