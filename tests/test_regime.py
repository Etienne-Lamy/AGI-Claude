"""Détection de régime en espace-champ : un module naît à la première transition,
la familiarité est un rappel ∈ [0,1], et l'identification est sans apprendre."""
import numpy as np
import torch

from scl.regime import DetecteurRegimeChamp


def _champ(rng, t=10):
    c = np.zeros((t, t), dtype=np.float32)
    c[t // 2, t // 2] = 0.25
    for _ in range(3):
        c[rng.integers(t), rng.integers(t)] = 1.0
    return c


def test_naissance_a_la_premiere_transition_et_familiarite_bornee():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    det = DetecteurRegimeChamp()
    # 1er champ : pas de transition (champ_prec None) → aucun module
    det.etape(_champ(rng))
    assert len(det.regimes) == 0
    # 2e champ : 1re transition → un module naît
    det.etape(_champ(rng))
    assert len(det.regimes) == 1
    # identifier ne crée pas et renvoie une familiarité dans [0,1]
    n = len(det.regimes)
    _, rap, fam = det.identifier(_champ(rng))
    assert len(det.regimes) == n
    assert 0.0 <= fam <= 1.0
    assert all(0.0 <= r <= 1.0 for r in rap.values())
