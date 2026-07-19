"""ÉTAPE 3 — l'orchestrateur naïf essaie un catalogue de dimensions et garde
celle de plus petit MDL (parcimonie §5). Test léger (petit catalogue, champs
synthétiques)."""
import numpy as np

from scl.orchestrateur_naif import essayer_catalogue


def _champ(rng, t=10):
    c = np.zeros((t, t), dtype=np.float32)
    c[t // 2, t // 2] = 0.25
    for _ in range(3):
        c[rng.integers(t), rng.integers(t)] = 1.0
    for _ in range(2):
        c[rng.integers(t), rng.integers(t)] = 0.5
    return c


def test_selection_par_mdl_renvoie_le_min():
    rng = np.random.default_rng(0)
    eval_ = [_champ(rng) for _ in range(15)]

    def flux(pas):
        r = np.random.default_rng(1)
        for _ in range(pas):
            yield _champ(r)

    meilleur, res = essayer_catalogue(flux, eval_, catalogue=[8, 32], pas=300)
    # structure attendue
    assert set(r["dim"] for r in res) == {8, 32}
    for r in res:
        assert r["mdl"] == round(r["code"] + r["residuel"], 1)
        assert r["code"] == r["dim"] * 0.5 or r["code"] == round(r["dim"] * 0.5, 1)
    # le meilleur est bien le MDL minimal du catalogue
    assert meilleur["mdl"] == min(r["mdl"] for r in res)
    assert "module" in meilleur
