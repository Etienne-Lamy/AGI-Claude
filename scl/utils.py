"""Utilitaires : ajustement de dimensions, projections déterministes, 2-means,
test séquentiel générique (SPRT)."""
import math

import torch

_PROJ_CACHE = {}


def ajuster_dim(v, n):
    """Aplatit et pad/tronque un vecteur à n dimensions (tolérance POC)."""
    if not isinstance(v, torch.Tensor):
        v = torch.as_tensor(v, dtype=torch.float32)
    v = v.flatten().float()
    if v.numel() == n:
        return v
    if v.numel() > n:
        return v[:n]
    return torch.cat([v, torch.zeros(n - v.numel())])


def projeter(v, n):
    """Projection aléatoire déterministe (cache par couple de dimensions).
    Sert à comparer/fusionner des flux de dimensions hétérogènes sans apprendre."""
    if not isinstance(v, torch.Tensor):
        v = torch.as_tensor(v, dtype=torch.float32)
    v = v.flatten().float().detach()
    key = (int(v.numel()), int(n))
    if key not in _PROJ_CACHE:
        g = torch.Generator().manual_seed(abs(hash(key)) % (2 ** 31))
        _PROJ_CACHE[key] = torch.randn(n, key[0], generator=g) / max(1, key[0]) ** 0.5
    return _PROJ_CACHE[key] @ v


def kmeans2(X, iters=25):
    """2-means minimal. X : tenseur (N, D). Retourne labels, c0, c1."""
    X = X.float()
    c0, c1 = X[0].clone(), X[-1].clone()
    labels = torch.zeros(X.shape[0], dtype=torch.long)
    for _ in range(iters):
        d0 = ((X - c0) ** 2).sum(1)
        d1 = ((X - c1) ** 2).sum(1)
        labels = (d1 < d0).long()
        if (labels == 0).any():
            c0 = X[labels == 0].mean(0)
        if (labels == 1).any():
            c1 = X[labels == 1].mean(0)
    return labels, c0, c1


def separation_claire(X, labels, c0, c1):
    """Vraie si la distance inter-centres domine la dispersion intra-cluster."""
    if (labels == 0).sum() < 3 or (labels == 1).sum() < 3:
        return False
    inter = float(((c0 - c1) ** 2).sum().sqrt())
    intra0 = float(((X[labels == 0] - c0) ** 2).sum(1).sqrt().mean())
    intra1 = float(((X[labels == 1] - c1) ** 2).sum(1).sqrt().mean())
    return inter > 1.5 * max(intra0, intra1, 1e-6)


def distance_contexte(a, b):
    """Distance générique entre deux contextes (tenseur ou scalaire) — sert
    à juger de la diversité d'un échantillon (disponibilité, création),
    jamais à apprendre."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        ta = a if isinstance(a, torch.Tensor) else torch.as_tensor(a, dtype=torch.float32)
        tb = b if isinstance(b, torch.Tensor) else torch.as_tensor(b, dtype=torch.float32)
        return float(((ta.flatten().float() - tb.flatten().float()) ** 2).sum().sqrt())
    return abs(float(a) - float(b))


def pente(valeurs):
    """Pente (régression linéaire simple) d'une série de scalaires."""
    n = len(valeurs)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(valeurs) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, valeurs))
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    return num / den


def limites_sprt(alpha, beta):
    """Bornes de décision du SPRT de Wald (Wald, 1945), en log-vraisemblance
    cumulée : Λ_n ≤ a → H0, Λ_n ≥ b → H1, sinon on continue."""
    a = math.log(beta / (1.0 - alpha))
    b = math.log((1.0 - beta) / alpha)
    return a, b


def sprt_sequentiel(increments, alpha=0.05, beta=0.10):
    """Test séquentiel de Wald générique (§4.3), réutilisé pour la surprise,
    la création et le drift (statistiques.py).

    `increments` : flux (itérable, dans l'ordre d'arrivée) d'incréments de
    log-rapport de vraisemblance log(p1(x_i)/p0(x_i)) — pas les observations
    brutes elles-mêmes, le calcul de p0/p1 est à la charge de l'appelant.
    Λ_n = Σ increments[:n], cumulée jusqu'au premier franchissement de borne.

    Retourne (decision, n) avec decision ∈ {"continuer", "H0", "H1"} et n le
    nombre d'incréments effectivement consommés (arrêt anticipé au
    franchissement, ou longueur totale du flux si aucune borne n'est
    atteinte)."""
    a, b = limites_sprt(alpha, beta)
    lam = 0.0
    n = 0
    for inc in increments:
        n += 1
        lam += float(inc)
        if lam <= a:
            return "H0", n
        if lam >= b:
            return "H1", n
    return "continuer", n
