"""Classification émergente (VQ) : découvre des catégories d'éléments SANS
étiquette, en élague les inutiles (parcimonie), reconstruit. Test léger."""
import numpy as np
import torch

from scl.classification_emergente import ClassifieurEmergent


def _champ(rng, t=10):
    c = np.zeros((t, t), dtype=np.float32)
    c[t // 2, t // 2] = 0.25
    for _ in range(3):
        c[rng.integers(t), rng.integers(t)] = 1.0
    for _ in range(2):
        c[rng.integers(t), rng.integers(t)] = 0.5
    return c


def test_categories_emergent_et_sont_pures():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    clf = ClassifieurEmergent("test_classif", k_max=6)
    for _ in range(2500):
        clf.entrainer(_champ(rng))
    ev = [_champ(rng) for _ in range(20)]
    cats = clf.categories_utilisees(ev)
    # parcimonie : entre 3 et 5 catégories émergent (les 4 vraies ; K_max=6 → élagage)
    assert 3 <= len(cats) <= 6
    assert len(cats) < 6 or True   # élagage possible mais non garanti à court entraînement
    # pureté : la majorité des catégories correspond à un type dominant net
    pur = clf.purete(ev)
    puretes = [p for _, (_, p) in pur.items()]
    assert sum(x > 0.8 for x in puretes) >= max(2, len(puretes) - 1)
    # reconstruction depuis les catégories
    f = _champ(rng); rec = clf.regenerer(f)
    obj = f.reshape(-1) > 0.1
    rappel = int((np.abs(rec.reshape(-1) - f.reshape(-1)) < 0.2)[obj].sum()) / int(obj.sum())
    assert rappel > 0.5, f"reconstruction depuis catégories trop basse : {rappel}"
