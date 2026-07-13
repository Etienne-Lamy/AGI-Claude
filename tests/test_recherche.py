"""Phase 8 — tests de recherche.py : A* dégénéré (V_ψ≡0) vérifié contre un
calcul à la main, TD sur V_ψ, refus d'ancrage sans point de vérité, A*
ancrée écartant les nœuds non ancrés."""
import torch

from scl.recherche import ValeurApprise, a_etoile, a_etoile_ancree, ancrer_composition, entrainer_v_psi

# graphe pondéré de référence — meilleur chemin A->B->C->D, coût 3
# (vérifié à la main : A->C->D coûte 5, A->B->D coûte 6)
_GRAPHE = {
    "A": [("B", 1), ("C", 4)],
    "B": [("C", 1), ("D", 5)],
    "C": [("D", 1)],
    "D": [],
}


def _voisins(n):
    return _GRAPHE[n]


# --------------------------------------------------------------------- ValeurApprise

def test_valeur_apprise_nulle_avant_tout_entrainement():
    v = ValeurApprise(dimension=1)
    assert v(torch.tensor([0.0])) == 0.0
    assert v(torch.tensor([5.0])) == 0.0


def test_entrainer_v_psi_deplace_la_valeur_vers_la_recompense_observee():
    torch.manual_seed(0)
    v = ValeurApprise(dimension=1)
    n, n_suivant = torch.tensor([0.0]), torch.tensor([1.0])
    for _ in range(100):
        entrainer_v_psi(v, n, r=1.0, n_suivant=n_suivant, gamma=0.0)   # gamma=0 : cible = r, constant
    assert v._entraine is True
    assert v(n) > 0.5   # converge vers la récompense observée (1.0)


# -------------------------------------------------------------------------- a_etoile

def test_a_etoile_degenere_recherche_exhaustive_sans_v_psi():
    chemin = a_etoile("A", "D", _voisins)
    assert chemin == ["A", "B", "C", "D"]


def test_a_etoile_degenere_avec_v_psi_jamais_entrainee():
    v = ValeurApprise(dimension=1)
    chemin = a_etoile("A", "D", _voisins, v_psi=lambda n: v(torch.zeros(1)))
    assert chemin == ["A", "B", "C", "D"]


def test_a_etoile_pas_de_chemin():
    graphe_isole = {"A": [], "B": []}
    assert a_etoile("A", "B", lambda n: graphe_isole.get(n, [])) is None


def test_a_etoile_objectif_predicat():
    chemin = a_etoile("A", lambda n: n == "C", _voisins)
    assert chemin[-1] == "C"


# --------------------------------------------------------------- ancrer_composition

def test_ancrer_composition_refuse_sans_point_de_verite():
    assert ancrer_composition() is None


def test_ancrer_composition_brut_seul():
    assert ancrer_composition(point_brut="reel_x") == ("brut", "reel_x")


def test_ancrer_composition_intermediaire_seul():
    assert ancrer_composition(module_certifie="mod_cert") == ("intermediaire", "mod_cert")


def test_ancrer_composition_arret_anticipe_si_intermediaire_suffit():
    niveau, point = ancrer_composition(point_brut="reel_x", module_certifie="mod_cert",
                                       erreur_intermediaire=0.05, erreur_brute_reference=0.05)
    assert niveau == "intermediaire"


def test_ancrer_composition_descend_au_brut_si_intermediaire_insuffisant():
    niveau, point = ancrer_composition(point_brut="reel_x", module_certifie="mod_cert",
                                       erreur_intermediaire=0.9, erreur_brute_reference=0.05)
    assert niveau == "brut"


# ---------------------------------------------------------------------- a_etoile_ancree

def test_a_etoile_ancree_ecarte_un_noeud_sans_point_de_verite():
    petit_graphe = {"A": [("B", 1)], "B": [("C", 1)], "C": []}

    def ancres(n):
        return None if n == "B" else ("brut", n)

    chemin = a_etoile_ancree("A", "C", lambda n: petit_graphe[n],
                             v_psi=lambda n: 0.0, ancres=ancres)
    assert chemin is None   # B, seul passage possible, n'est jamais ancré


def test_a_etoile_ancree_trouve_si_tout_est_ancre():
    petit_graphe = {"A": [("B", 1)], "B": [("C", 1)], "C": []}
    chemin = a_etoile_ancree("A", "C", lambda n: petit_graphe[n],
                             v_psi=lambda n: 0.0, ancres=lambda n: ("brut", n))
    assert chemin == ["A", "B", "C"]
