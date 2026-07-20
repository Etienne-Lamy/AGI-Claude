"""Mémoire épisodique : l'enregistreur scelle un épisode quand la surprise se
termine ; la mémoire ne garde que le surprenant et purge les COMPRIS d'abord."""
import numpy as np

from scl.memoire_episodique import Enregistreur, MemoireEpisodique, Episode
from scl.config import CONFIG


def _champ(v=1.0):
    c = np.zeros((10, 10), dtype=np.float32); c[5, 5] = 0.25; c[v_row(v), 3] = v
    return c
def v_row(v): return 2 if v == 1.0 else 4


def test_enregistreur_scelle_a_la_fin_de_surprise():
    enr = Enregistreur(seuil_surprise=0.35)
    # pas surpris → rien
    assert enr.observer(_champ(), (0, 0), "r0", familiarite=0.9) is None
    # surprise prolongée → accumulation, rien scellé
    for _ in range(6):
        assert enr.observer(_champ(), (1, 0), None, familiarite=0.1) is None
    # retour à familier → l'épisode se scelle
    ep = enr.observer(_champ(), (0, 0), "r0", familiarite=0.9)
    assert ep is not None
    assert len(ep.champs) == 6 and ep.familiarite_min <= 0.1


def test_memoire_purge_les_compris_en_priorite():
    m = MemoireEpisodique(capacite=2)
    e1 = Episode(_champ(), [], [], [_champ()], 0.1, compris=True)
    e2 = Episode(_champ(), [], [], [_champ()], 0.1, compris=False)
    m.enregistrer(e1); m.enregistrer(e2)
    e3 = Episode(_champ(), [], [], [_champ()], 0.1, compris=False)
    m.enregistrer(e3)                      # dépasse la capacité → purge un COMPRIS (e1)
    ids = [id(e) for e in m.episodes]
    assert id(e1) not in ids and id(e2) in ids and id(e3) in ids
