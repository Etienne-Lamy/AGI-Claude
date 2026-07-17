"""Tests du modèle de prévision du corps (prevision.py) : apprentissage en
ligne de la dynamique (v, accel) → v', montée de la fiabilité, et robustesse
du transfert instinct → appris."""
from scl.config import CONFIG
from scl.prevision import ModelePrevisionCorps


def test_fiabilite_nulle_avant_apprentissage():
    """Tant qu'il n'a pas appris, le modèle ne mérite aucune confiance :
    l'instinct doit garder la main (dégénérescence, §1.4)."""
    m = ModelePrevisionCorps(v_max=2)
    assert m.fiabilite() == 0.0


def test_apprend_la_dynamique_du_corps():
    """Après entraînement sur la vraie dynamique v' = clip(v+accel), le modèle
    la prédit et sa fiabilité franchit le seuil de bascule."""
    m = ModelePrevisionCorps(v_max=2)
    accels = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]
    # entraînement sur des transitions vraies
    for _ in range(400):
        for v in [(0, 0), (1, 0), (2, 1), (-2, -1), (1, -2), (2, 2)]:
            for a in accels:
                vp = (max(-2, min(2, v[0] + a[0])), max(-2, min(2, v[1] + a[1])))
                m.apprendre(v, a, vp)
    # prédiction correcte incluant la saturation (2+1 clampé à 2)
    assert m.predire((2, 0), (1, 0)) == (2, 0)
    assert m.predire((0, 0), (0, 1)) == (0, 1)
    assert m.predire((-2, 0), (-1, 0)) == (-2, 0)
    assert m.fiabilite() >= CONFIG["seuil_fiabilite_appris"]


def test_predire_borne_a_vmax():
    m = ModelePrevisionCorps(v_max=2)
    vp = m.predire((0, 0), (1, 0))
    assert -2 <= vp[0] <= 2 and -2 <= vp[1] <= 2
