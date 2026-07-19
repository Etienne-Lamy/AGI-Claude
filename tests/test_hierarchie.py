"""Niveau N3 : (régime, action) → régime suivant. Le module doit apprendre une
règle de transition déterministe et battre le prior trivial « rien ne change »."""
import torch

from scl.hierarchie import ModuleTransitionRegime, gain_vs_trivial


def test_gain_vs_trivial_se_calcule():
    # trivial parfait (rien ne change) ⇒ gain nul quel que soit le prédicteur
    just, triv, gain, n_chg = gain_vs_trivial([0, 1], [0, 1], [0, 1])
    assert just == 1.0 and triv == 1.0 and n_chg == 0
    # trivial se trompe partout, prédicteur juste partout ⇒ gain maximal
    just, triv, gain, n_chg = gain_vs_trivial([1, 0], [1, 0], [0, 1])
    assert just == 1.0 and triv == 0.0 and gain == 1.0 and n_chg == 2


def test_apprend_une_regle_de_transition_deterministe():
    """Règle : action 1 fait +1 (borné), action 2 fait −1, action 0 ne change rien.
    C'est la forme abstraite de « l'accélération change le régime de vitesse »."""
    torch.manual_seed(0)
    n_reg, n_act = 5, 3
    n3 = ModuleTransitionRegime("t", n_reg, n_act)

    def suivant(r, a):
        if a == 1:
            return min(n_reg - 1, r + 1)
        if a == 2:
            return max(0, r - 1)
        return r

    for _ in range(400):
        for r in range(n_reg):
            for a in range(n_act):
                n3.entrainer(r, a, suivant(r, a))

    preds, verites, precedents = [], [], []
    for r in range(n_reg):
        for a in range(n_act):
            preds.append(n3.predire(r, a)); verites.append(suivant(r, a)); precedents.append(r)
    just, triv, gain, n_chg = gain_vs_trivial(preds, verites, precedents)
    assert just > 0.8, f"n'a pas appris la règle : {just}"
    assert gain > 0.5, f"n'apporte rien sur le prior trivial : {gain}"


def test_table_est_lisible():
    n3 = ModuleTransitionRegime("t", 3, 2)
    t = n3.table(noms_regimes=["a", "b", "c"], noms_actions=["rien", "plus"])
    assert set(t) == {(r, a) for r in "abc" for a in ["rien", "plus"]}
    assert all(v in ("a", "b", "c") for v in t.values())
