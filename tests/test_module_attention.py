"""Outil attention/masquage (Slot Attention) : décompose le champ en objets,
reconstruit, et fournit une liste d'objets (latent structuré). Test léger."""
import numpy as np

from scl.module_attention import ModuleAttentionSlots


def _champ(rng, t=10):
    c = np.zeros((t, t), dtype=np.float32)
    c[t // 2, t // 2] = 0.25
    for _ in range(3):
        c[rng.integers(t), rng.integers(t)] = 1.0
    for _ in range(2):
        c[rng.integers(t), rng.integers(t)] = 0.5
    return c


def test_slot_attention_reconstruit_et_extrait_des_objets():
    rng = np.random.default_rng(0)
    mod = ModuleAttentionSlots("test_attn", n_slots=8, D=64)
    for _ in range(1500):
        mod.entrainer(_champ(rng))
    fids = [mod.fidelite(_champ(rng)) for _ in range(15)]
    rappel = sum(d["rappel"] for d in fids) / len(fids)
    assert rappel > 0.55, f"reconstruction objet-centrée trop basse : {rappel}"
    # la liste d'objets (latent structuré) n'est pas vide sur un champ peuplé
    objets = mod.liste_objets(_champ(rng))
    assert len(objets) > 0
    for (i, j, typ) in objets:                    # (row, col, type) valides
        assert 0 <= i < 10 and 0 <= j < 10 and typ in (1, 2, 3)
