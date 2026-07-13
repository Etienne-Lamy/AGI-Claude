"""Recherche de composition SCL — A*, heuristique apprise V_ψ, ancrage à un
point de vérité vérifiable, A* ancrée (§7).

`a_etoile` dégénère en recherche non informée quand V_ψ≡0 (historique vide,
§7.3) : ce n'est PAS un raccourci, c'est un comportement explicitement
sanctionné par la théorie — une nouvelle instance de `ValeurApprise` n'a
jamais été entraînée, donc renvoie 0.0 partout, et A* se comporte alors
exactement comme une recherche à coût uniforme (Dijkstra)."""
import heapq
import itertools

import torch

from .config import CONFIG
from .graphe import test_non_inferiorite
from .logger import log, log_verbeux
from .utils import ajuster_dim


class ValeurApprise:
    """V_ψ (§7.2) : heuristique apprise, NON garantie admissible — best-first
    pondéré (Pohl, 1970). Tant qu'aucune mise à jour TD n'a eu lieu, renvoie
    identiquement 0.0 (dégénérescence exhaustive, §7.3)."""

    def __init__(self, dimension=None):
        self.dimension = dimension or CONFIG["dim_emb"]
        h = CONFIG["n_hidden_v_psi"]
        self.W1 = torch.nn.Parameter(torch.randn(h, self.dimension) * (1.0 / max(1, self.dimension)) ** 0.5)
        self.b1 = torch.nn.Parameter(torch.zeros(h))
        self.W2 = torch.nn.Parameter(torch.randn(1, h) * (1.0 / max(1, h)) ** 0.5)
        self.b2 = torch.nn.Parameter(torch.zeros(1))
        self._g = [torch.zeros_like(p) for p in self.parametres()]
        self._entraine = False

    def parametres(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def _forward(self, n):
        x = ajuster_dim(n, self.dimension)
        h = torch.relu(self.W1 @ x + self.b1)
        return (self.W2 @ h + self.b2).squeeze(0)

    def __call__(self, n):
        if not self._entraine:
            return 0.0
        with torch.no_grad():
            return float(self._forward(n))


def v_psi(instance, n):
    """Enrobage fonctionnel de `ValeurApprise.__call__`, pour matcher la
    signature cible "nœud → valeur" indépendamment de l'implémentation."""
    return instance(n)


def entrainer_v_psi(v, n, r, n_suivant, gamma=None, phase="jour"):
    """Mise à jour TD (Robbins-Monro, 1951), §7.2 :
    ψ ← ψ + α(r + γV_ψ(n') - V_ψ(n))∇_ψV_ψ(n). Dès le premier appel, V_ψ
    n'est plus ≡0 (fin de la dégénérescence exhaustive)."""
    gamma = gamma if gamma is not None else CONFIG["gamma_v_psi"]
    v._entraine = True
    for p in v.parametres():
        p.grad = None
    valeur_n = v._forward(n)
    with torch.no_grad():
        cible = r + gamma * v._forward(n_suivant)
    erreur_td = (cible - valeur_n) ** 2
    erreur_td.backward()
    torch.nn.utils.clip_grad_norm_(v.parametres(), CONFIG["clip_grad_simulateur"])
    beta = CONFIG["beta_jour"] if phase == "jour" else CONFIG["beta_nuit"]
    grads = [p.grad for p in v.parametres()]
    for i, g in enumerate(grads):
        if g is not None:
            v._g[i].mul_(beta).add_(g, alpha=1 - beta)
    with torch.no_grad():
        for p, g in zip(v.parametres(), v._g):
            p -= CONFIG["lr_v_psi"] * g
    e = float(erreur_td.detach())
    log_verbeux("recherche", "entrainement_v_psi", erreur_td=e, phase=phase)
    return e


def a_etoile(depart, objectif, voisins, v_psi=None, budget_max=None):
    """A* : f(n)=g(n)+h(n). `voisins` : nœud → [(nœud_suivant, coût_arête)].
    `objectif` : nœud, ou prédicat nœud→bool. `v_psi` : nœud → valeur
    heuristique (None ⇒ 0 partout ⇒ dégénérescence exhaustive, §7.3).
    Retourne le chemin (liste de nœuds) ou None si aucun trouvé."""
    v_psi = v_psi or (lambda n: 0.0)
    test_objectif = objectif if callable(objectif) else (lambda n: n == objectif)
    compteur = itertools.count()   # départage stable (évite de comparer les nœuds entre eux)
    frontiere = [(v_psi(depart), next(compteur), depart, 0.0, [depart])]
    visites = {}
    while frontiere:
        f, _, n, g, chemin = heapq.heappop(frontiere)
        if budget_max is not None and len(visites) >= budget_max:
            log_verbeux("recherche", "a_etoile_budget_epuise", budget_max=budget_max)
            return None
        if n in visites and visites[n] <= g:
            continue
        visites[n] = g
        if test_objectif(n):
            log_verbeux("recherche", "a_etoile_trouve", longueur=len(chemin), cout=g)
            return chemin
        for suivant, cout_arete in voisins(n):
            g2 = g + cout_arete
            if suivant in visites and visites[suivant] <= g2:
                continue
            f2 = g2 + v_psi(suivant)
            heapq.heappush(frontiere, (f2, next(compteur), suivant, g2, chemin + [suivant]))
    log_verbeux("recherche", "a_etoile_echec")
    return None


def ancrer_composition(point_brut=None, module_certifie=None,
                       erreur_intermediaire=None, erreur_brute_reference=None):
    """Détermine le point de comparaison VÉRIFIABLE d'une composition
    candidate (§7.4, §0) : (A) le réel brut si la prédiction est descendue
    jusque-là (`point_brut`), (B) un module déjà certifié si suffisant
    (`module_certifie`) — arrêt anticipé via `test_non_inferiorite` : ne
    redescend au brut que si le module intermédiaire ne suffit pas.

    Sans AUCUN point de vérité disponible : refuse (None) — un résidu non
    mesurable ne peut recevoir aucun signal d'apprentissage.

    Retourne (niveau, point) avec niveau ∈ {"brut", "intermediaire"}, ou None."""
    if point_brut is None and module_certifie is None:
        log("recherche", "ancrer_composition_refusee", raison="aucun_point_de_verite")
        return None
    if module_certifie is not None:
        if point_brut is None:
            log_verbeux("recherche", "ancrer_composition", niveau="intermediaire")
            return "intermediaire", module_certifie
        if erreur_intermediaire is not None and erreur_brute_reference is not None:
            if test_non_inferiorite([erreur_brute_reference], [erreur_intermediaire]):
                log("recherche", "ancrer_composition", niveau="intermediaire",
                    arret_anticipe=True)
                return "intermediaire", module_certifie
    log("recherche", "ancrer_composition", niveau="brut")
    return "brut", point_brut


def a_etoile_ancree(depart, objectif, voisins, v_psi, ancres, profondeur_max=None):
    """A* dont g(n)/h(n) sont évalués via le point d'ancrage disponible à
    chaque nœud (`ancrer_composition`, §7.4) — fusion réel/imaginé pondérée
    par la profondeur (confiance décroissante avec la profondeur, §15.1 ;
    ici une pondération 1/(1+profondeur) — [D] simplification documentée en
    attendant `decision_action.fusion_ponderee`, Phase 10).

    `ancres` : nœud → résultat de `ancrer_composition` (ou None). Un nœud
    sans ancrage est écarté : aucun signal, aucune décision fiable (§7.4)."""
    profondeur_max = profondeur_max or CONFIG["profondeur_max_recherche"]
    test_objectif = objectif if callable(objectif) else (lambda n: n == objectif)
    compteur = itertools.count()
    frontiere = [(v_psi(depart), next(compteur), depart, 0.0, [depart], 0)]
    visites = {}
    while frontiere:
        f, _, n, g, chemin, profondeur = heapq.heappop(frontiere)
        if n in visites and visites[n] <= g:
            continue
        visites[n] = g
        ancre = ancres(n) if ancres is not None else None
        if ancre is None:
            continue   # §7.4 : pas de point de vérité, nœud écarté
        if test_objectif(n):
            return chemin
        if profondeur >= profondeur_max:
            continue
        confiance = 1.0 / (1.0 + profondeur)
        for suivant, cout_arete in voisins(n):
            g2 = g + cout_arete * (2.0 - confiance)
            if suivant in visites and visites[suivant] <= g2:
                continue
            f2 = g2 + v_psi(suivant)
            heapq.heappush(frontiere,
                           (f2, next(compteur), suivant, g2, chemin + [suivant], profondeur + 1))
    return None
