"""ÉTAPE 1 — tests de l'autoencodeur de vision (module_ae.py) : il reconstruit
vraiment le champ (pas d'effondrement à zéro), et le champ abstrait est de la
bonne forme. Entraînement court sur des champs synthétiques sparses."""
import numpy as np

from scl.config import CONFIG
from scl.module_ae import ModuleAutoencodeur


def _champ_aleatoire(rng, t=10, n_sucre=3, n_baton=2):
    champ = np.zeros((t, t), dtype=np.float32)
    champ[t // 2, t // 2] = 0.25                      # corps au centre
    for _ in range(n_sucre):
        champ[rng.integers(t), rng.integers(t)] = 1.0
    for _ in range(n_baton):
        champ[rng.integers(t), rng.integers(t)] = 0.5
    return champ


def test_reconstruit_sans_seffondrer():
    """Après quelques centaines de pas, l'autoencodeur reconstruit la majorité
    des objets (rappel élevé) ET n'hallucine pas (précision élevée) — donc pas
    l'effondrement à zéro (qui donnerait 0 objet reconstruit)."""
    rng = np.random.default_rng(0)
    ae = ModuleAutoencodeur("test_vision")
    for _ in range(800):
        ae.entrainer(_champ_aleatoire(rng))
    # évalue sur des champs frais
    rappels, precisions, reconstruits = [], [], []
    for _ in range(20):
        f = _champ_aleatoire(rng)
        d = ae.fidelite(f)
        rappels.append(d["rappel"]); precisions.append(d["precision"])
        reconstruits.append(d["n_objets_reconstruits"])
    rappel = sum(rappels) / len(rappels)
    precision = sum(precisions) / len(precisions)
    assert sum(reconstruits) > 0, "effondrement : rien n'est reconstruit"
    assert rappel > 0.8, f"rappel trop bas : {rappel}"
    assert precision > 0.8, f"précision trop basse : {precision}"


def test_champ_abstrait_bonne_forme():
    ae = ModuleAutoencodeur("test_vision")
    t = CONFIG["taille_perception"]
    z = ae.encoder(_champ_aleatoire(np.random.default_rng(1)))
    assert tuple(z.shape) == (ae.canaux_latent, t, t)   # champ abstrait spatial


def test_incertitude_descend_avec_l_apprentissage():
    rng = np.random.default_rng(2)
    ae = ModuleAutoencodeur("test_vision")
    for _ in range(30):
        ae.entrainer(_champ_aleatoire(rng))
    inc_debut = ae.incertitude()
    for _ in range(600):
        ae.entrainer(_champ_aleatoire(rng))
    assert ae.incertitude() < inc_debut
