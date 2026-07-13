"""Phase 0 — tests du socle : utils.py (dont le nouveau sprt_sequentiel)."""
import math
import random

import torch

from scl.utils import (
    ajuster_dim, projeter, kmeans2, separation_claire, pente,
    limites_sprt, sprt_sequentiel,
)


# --------------------------------------------------------------------- SPRT

def _increment_bernoulli(x, p0, p1):
    """log(p1(x)/p0(x)) pour x binaire, sous deux Bernoulli p0 et p1."""
    lp0 = math.log(p0) if x else math.log(1 - p0)
    lp1 = math.log(p1) if x else math.log(1 - p1)
    return lp1 - lp0


def test_limites_sprt_formule():
    a, b = limites_sprt(alpha=0.05, beta=0.10)
    assert abs(a - math.log(0.10 / 0.95)) < 1e-9
    assert abs(b - math.log(0.90 / 0.05)) < 1e-9
    assert a < 0 < b   # H0 en dessous, H1 au-dessus, zone médiane = "continuer"


def test_sprt_conclut_h1_sur_evidence_forte():
    # H0: p=0.1 (rare), H1: p=0.9 (fréquent) ; on observe x=1 en rafale.
    increments = [_increment_bernoulli(1, p0=0.1, p1=0.9) for _ in range(50)]
    decision, n = sprt_sequentiel(increments, alpha=0.05, beta=0.10)
    assert decision == "H1"
    assert n < 50   # arrêt anticipé, pas besoin de tout le flux


def test_sprt_conclut_h0_sur_evidence_forte():
    increments = [_increment_bernoulli(0, p0=0.1, p1=0.9) for _ in range(50)]
    decision, n = sprt_sequentiel(increments, alpha=0.05, beta=0.10)
    assert decision == "H0"
    assert n < 50


def test_sprt_continue_sur_evidence_ambigue():
    # deux observations à peine informatives : ne doit pas trancher.
    increments = [_increment_bernoulli(1, p0=0.5, p1=0.55) for _ in range(2)]
    decision, n = sprt_sequentiel(increments, alpha=0.05, beta=0.10)
    assert decision == "continuer"
    assert n == 2


def test_sprt_flux_vide():
    decision, n = sprt_sequentiel([], alpha=0.05, beta=0.10)
    assert (decision, n) == ("continuer", 0)


def test_sprt_sous_h0_conclut_rarement_h1():
    # flux généré sous la loi nulle (p=0.5 des deux côtés testés à 0.3/0.7) :
    # sur un run typique, le SPRT doit conclure H0 (ou continuer), pas H1.
    rng = random.Random(0)
    increments = [
        _increment_bernoulli(1 if rng.random() < 0.5 else 0, p0=0.3, p1=0.7)
        for _ in range(200)
    ]
    decision, _ = sprt_sequentiel(increments, alpha=0.05, beta=0.10)
    assert decision != "H1"


# ------------------------------------------------------- fonctions reprises

def test_ajuster_dim_pad():
    v = torch.tensor([1.0, 2.0])
    out = ajuster_dim(v, 5)
    assert out.shape == (5,)
    assert out[:2].tolist() == [1.0, 2.0]
    assert out[2:].tolist() == [0.0, 0.0, 0.0]


def test_ajuster_dim_tronque():
    v = torch.tensor([1.0, 2.0, 3.0, 4.0])
    out = ajuster_dim(v, 2)
    assert out.tolist() == [1.0, 2.0]


def test_projeter_deterministe():
    v = torch.randn(6)
    p1 = projeter(v, 3)
    p2 = projeter(v, 3)
    assert torch.equal(p1, p2)   # cache déterministe, même clé -> même matrice
    assert p1.shape == (3,)


def test_kmeans2_separation_claire():
    torch.manual_seed(0)
    a = torch.randn(10, 2) + torch.tensor([5.0, 5.0])
    b = torch.randn(10, 2) + torch.tensor([-5.0, -5.0])
    X = torch.cat([a, b])
    labels, c0, c1 = kmeans2(X)
    assert separation_claire(X, labels, c0, c1)


def test_pente_croissante_decroissante():
    assert pente([1, 2, 3, 4, 5]) > 0
    assert pente([5, 4, 3, 2, 1]) < 0
    assert abs(pente([3, 3, 3, 3])) < 1e-9
