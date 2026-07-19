"""Outil attention/masquage (Slot Attention) : décompose le champ en objets,
reconstruit, et fournit une liste d'objets (latent structuré). Test léger."""
import numpy as np
import torch

from scl.module_attention import ModuleAttentionSlots


def _champ(rng, t=10):
    c = np.zeros((t, t), dtype=np.float32)
    c[t // 2, t // 2] = 0.25
    for _ in range(3):
        c[rng.integers(t), rng.integers(t)] = 1.0
    for _ in range(2):
        c[rng.integers(t), rng.integers(t)] = 0.5
    return c


def test_slot_attention_plomberie():
    """Test de PLOMBERIE (pas de convergence) : slot attention est un module
    stochastique lourd et variable en entraînement court + non-déterminisme CUDA ;
    sa PERFORMANCE réelle (94 % rappel) est validée par le harnais `etape4_attention`
    (reproductible). Ici on garde seulement contre les régressions d'API/formes et
    l'effondrement dur : l'entraînement tourne, encodage/reconstruction/liste
    d'objets renvoient les bonnes formes/types, l'erreur descend."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    mod = ModuleAttentionSlots("test_attn", n_slots=8, D=64)
    inc0 = None
    for i in range(400):
        mod.entrainer(_champ(rng))
        if i == 20:
            inc0 = mod.incertitude()
    assert mod.incertitude() <= inc0                       # l'erreur ne diverge pas
    champ = _champ(rng)
    rec = mod.reconstruire(champ)
    assert tuple(rec.shape) == (100,)                       # champ reconstruit (t*t)
    z = mod.encoder(champ)
    assert tuple(z.shape) == (mod.n_slots, mod.D)           # slots (latent objet)
    for (i, j, typ) in mod.liste_objets(champ):             # liste d'objets valide
        assert 0 <= i < 10 and 0 <= j < 10 and typ in (1, 2, 3)
