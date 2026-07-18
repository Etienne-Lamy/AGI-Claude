"""Tests de l'émergence de la dynamique (dynamique.py) : un prédicteur naît
d'une surprise confirmée sur une accélération qui change vraiment la vitesse ;
l'accélération nulle (rien ne change) n'en fait jamais naître ; l'incertitude
d'une action guide la curiosité."""
from scl.dynamique import Dynamique
from scl.config import CONFIG


def test_accel_nulle_ne_cree_aucun_predicteur():
    """Le prior « rien ne change » est exact pour (0,0) : aucune surprise,
    aucun module (curiosité trivialement satisfaite)."""
    d = Dynamique()
    for _ in range(60):
        d.observer((0, 0), (0, 0), (0, 0))     # v inchangée
    assert (0, 0) not in d.predicteurs
    assert len(d.predicteurs) == 0


def _grille_vitesses():
    return [(vx, vy) for vx in range(-2, 3) for vy in range(-2, 3)]


def test_accel_qui_bouge_fait_naitre_un_predicteur():
    """Une accélération qui change réellement la vitesse (surprise vs le prior)
    finit par faire naître un prédicteur dédié, une fois assez de contextes
    (vitesses) DISTINCTS accumulés (SPRT confirmé, §4.5)."""
    d = Dynamique()
    for _ in range(10):
        for v in _grille_vitesses():           # 25 vitesses distinctes
            vx = min(2, v[0] + 1)
            d.observer(v, (1, 0), (vx, v[1]))   # +x : v -> v+(1,0) borné
    assert (1, 0) in d.predicteurs


def test_incertitude_action_inexploree_est_l_attrait():
    d = Dynamique()
    assert d.incertitude_action((0, 0), (0, 1)) == CONFIG["attrait_action_inexploree"]


def test_predicteur_apprend_baisse_l_incertitude():
    """Après création, entraîner le prédicteur sur sa dynamique fait baisser
    son incertitude (progrès d'apprentissage)."""
    d = Dynamique()
    for _ in range(20):
        for v in _grille_vitesses():
            vx = min(2, v[0] + 1)
            d.observer(v, (1, 0), (vx, v[1]))
    assert (1, 0) in d.predicteurs
    inc_tot = d.incertitude_action((0, 0), (1, 0))
    # l'incertitude du prédicteur entraîné reste finie et raisonnable
    assert inc_tot < CONFIG["attrait_action_inexploree"] * 5
