"""Pulsions (étape 17) : faim/douleur de corps, pulsions cognitives, objectif dominant
unique (argmax+hystérésis) avec réflexe douleur prioritaire, récompense pour la planif."""
from scl.pulsions import Pulsions
from scl.config import CONFIG


def test_faim_monte_puis_chute_sur_sucre():
    p = Pulsions()
    f0 = p.besoins.etats["faim"]
    p.maj()
    assert p.besoins.etats["faim"] > f0            # la faim creuse avec le temps
    p.maj(evenements=["sucre"])
    assert p.besoins.etats["faim"] < f0            # manger la fait chuter


def test_douleur_declenche_le_reflexe_prioritaire():
    p = Pulsions()
    p.maj(curiosite=0.9)                            # une pulsion cognitive forte…
    p.maj(evenements=["baton"], curiosite=0.9)     # …mais un choc doit tout court-circuiter
    assert p.douleur > CONFIG["seuil_reflexe_douleur"]
    assert p.objectif_dominant() == "douleur"


def test_douleur_persiste_puis_retombe():
    p = Pulsions()
    p.maj(evenements=["baton"])
    d1 = p.douleur
    for _ in range(3):
        p.maj()                                    # décroissance soustractive lente
    assert 0.0 <= p.douleur < d1                    # persiste mais retombe


def test_objectif_dominant_suit_la_pulsion_forte():
    p = Pulsions()
    for _ in range(5):
        p.maj(curiosite=0.9)                        # curiosité domine nettement
    assert p.objectif_dominant() == "curiosite"


def test_recompense_signee():
    p = Pulsions()
    assert p.recompense(evenements=["sucre"]) > 0
    assert p.recompense(evenements=["baton"]) < 0
    assert p.recompense() < 0                       # temps perdu : léger négatif
    assert p.recompense(progres=1.0) > p.recompense()   # le progrès récompense
