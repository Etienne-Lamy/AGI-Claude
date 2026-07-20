"""Auto-réglage §28.4 : la boucle garde un changement d'hyperparamètre SEULEMENT s'il
améliore l'observable (réversible), et converge vers l'optimum local."""
from scl.autoreglage import AutoReglage, SYMPTOMES


def test_optimise_vers_le_maximum_de_lobservable():
    # observable synthétique maximal en theta=3 ; départ à 0, pas ±1 → doit atteindre 3.
    etat = {"theta": 0}
    def appliquer(v): etat["theta"] = v
    def mesurer():    return -((etat["theta"] - 3) ** 2)   # plus grand = mieux, max en 3
    r = AutoReglage()
    final = r.optimiser("theta", 0, deltas=(-1, +1), appliquer=appliquer, mesurer=mesurer)
    assert final == 3
    assert etat["theta"] == 3          # l'état système est bien posé sur l'optimum


def test_reversible_ne_garde_pas_un_changement_qui_degrade():
    # à l'optimum, aucun voisin n'améliore → la valeur ne bouge pas (revert systématique).
    etat = {"theta": 3}
    def appliquer(v): etat["theta"] = v
    def mesurer():    return -((etat["theta"] - 3) ** 2)
    r = AutoReglage()
    v, s = r.regler("theta", 3, deltas=(-1, +1), appliquer=appliquer, mesurer=mesurer)
    assert v == 3 and s == 0
    assert etat["theta"] == 3          # remis à l'état d'avant, pas laissé sur un voisin pire
    assert r.historique[-1]["garde"] is False


def test_marge_evite_de_courir_apres_le_bruit():
    # une amélioration plus petite que la marge n'est PAS gardée (asymétrie §28.4).
    etat = {"theta": 0}
    def appliquer(v): etat["theta"] = v
    def mesurer():    return 0.0 if etat["theta"] == 0 else 1e-9   # gain infime en bougeant
    r = AutoReglage(marge=1e-3)
    v, _ = r.regler("theta", 0, deltas=(-1, +1), appliquer=appliquer, mesurer=mesurer)
    assert v == 0                      # gain < marge → on ne bouge pas


def test_correctif_pour_traduit_un_diagnostic():
    assert SYMPTOMES  # non vide
    param, sens = AutoReglage().correctif_pour("sur_creation_modules")
    assert param == "grace_regime" and sens == +1
    assert AutoReglage().correctif_pour("diagnostic_inconnu") is None
