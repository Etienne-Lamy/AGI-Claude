"""Tests de la motivation intrinsèque (curiosite.py) : incertitude, progrès
d'apprentissage, maîtrise, frontière."""
from scl import curiosite
from scl.config import CONFIG


class _FauxModule:
    """Module minimal : seul `error_history` (liste de (contexte, erreur, t))
    est requis par curiosite.py."""
    def __init__(self, erreurs):
        self.error_history = [(None, e, i) for i, e in enumerate(erreurs)]


def test_incertitude_module_jamais_evalue_est_maximale():
    m = _FauxModule([])
    assert curiosite.incertitude(m) == CONFIG["incertitude_initiale"]


def test_incertitude_suit_l_erreur_recente():
    m = _FauxModule([0.5] * 10 + [0.01] * 20)   # récemment très bas
    assert curiosite.incertitude(m, fenetre=20) < 0.02


def test_progres_positif_quand_l_erreur_descend():
    # ancienne fenêtre haute, récente basse ⇒ progrès > 0
    m = _FauxModule([0.5] * 20 + [0.1] * 20)
    assert curiosite.progres_apprentissage(m, fenetre=20) > 0


def test_progres_nul_sans_recul_suffisant():
    m = _FauxModule([0.3] * 5)
    assert curiosite.progres_apprentissage(m, fenetre=20) == 0.0


def test_maitrise_quand_bas_et_plat():
    # beaucoup de vécu, incertitude basse, plus de progrès
    m = _FauxModule([0.01] * 80)
    assert curiosite.maitrise(m)


def test_pas_maitrise_si_encore_haut():
    m = _FauxModule([0.5] * 80)
    assert not curiosite.maitrise(m)


def test_frontiere_choisit_le_moins_maitrise():
    maitrise = _FauxModule([0.005] * 80)      # maîtrisé
    a_apprendre = _FauxModule([0.4] * 80)     # incertain, pas maîtrisé
    modules = {"maitrise": maitrise, "frontiere": a_apprendre}
    assert curiosite.frontiere(modules) == "frontiere"


def test_frontiere_none_si_tout_maitrise():
    modules = {"a": _FauxModule([0.005] * 80), "b": _FauxModule([0.008] * 80)}
    assert curiosite.frontiere(modules) is None
