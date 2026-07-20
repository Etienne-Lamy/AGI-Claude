"""ORCHESTRATEUR — Mode B « appris » (§31.6) : émetteur de programmes autorégressif.

L'orchestrateur pensé comme un LLM à attention (vision de l'auteur) : au lieu de
CHERCHER (Mode A, coûteux car il entraîne chaque programme), il ÉMET directement le
programme, token par token, conditionné par l'OBJECTIF (le besoin qui sélectionne la
cible, §31.4). Émission TYPÉE : à chaque pas, seules les continuations bien typées
sont émettables (masque dur, §31.3). Appris par IMITATION des programmes gagnants de
Mode A (démarrage à froid, analogue au précâblage à la naissance) ; REINFORCE ensuite
(non fait ici).

Intérêt démontré : le meilleur programme DÉPEND de l'objectif (prédire ≠ reconstruire) ;
Mode B apprend cette dépendance au contexte et émet le bon programme SANS refaire la
recherche — il AMORTIT le coût de Mode A.
"""
import torch

from .logger import log
from .module_ae import DEVICE
from .orchestrateur import OPERATEURS

# vocabulaire d'émission = opérateurs + EOF ; BOS pour amorcer
VOCAB = list(OPERATEURS) + ["EOF"]
BOS = len(VOCAB)


def _type_apres(chaine):
    """Type du signal courant après la chaîne d'opérateurs émise (part de 'champ')."""
    t = "champ"
    for op in chaine:
        t = OPERATEURS[op][1]
    return t


def _masque_type(chaine):
    """Booléens sur VOCAB : opérateurs typables depuis l'état courant + EOF si on est
    revenu au type 'champ' (programme terminal valide)."""
    t = _type_apres(chaine)
    m = torch.zeros(len(VOCAB), dtype=torch.bool)
    for i, tok in enumerate(VOCAB):
        if tok == "EOF":
            m[i] = (t == "champ" and len(chaine) >= 1)
        else:
            m[i] = (OPERATEURS[tok][0] == t)
    return m


class ModeB(torch.nn.Module):
    """GRU autorégressif conditionné par l'objectif, à émission typée."""

    def __init__(self, n_objectifs, dim=64):
        super().__init__()
        self.emb_obj = torch.nn.Embedding(n_objectifs, dim)
        self.emb_tok = torch.nn.Embedding(len(VOCAB) + 1, dim)   # +1 pour BOS
        self.gru = torch.nn.GRUCell(dim, dim)
        self.tete = torch.nn.Linear(dim, len(VOCAB))
        self.dim = dim
        self.to(DEVICE)

    def _logits(self, h):
        return self.tete(h)

    def emettre(self, objectif, max_len=4):
        """Génère un programme (greedy) pour l'objectif, sous masque de type."""
        h = self.emb_obj(torch.tensor([objectif], device=DEVICE)).squeeze(0)
        tok = torch.tensor(BOS, device=DEVICE)
        chaine = []
        for _ in range(max_len):
            h = self.gru(self.emb_tok(tok).unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            logits = self._logits(h)
            masque = _masque_type(chaine).to(DEVICE)
            logits = logits.masked_fill(~masque, float("-inf"))
            tok = torch.argmax(logits)
            if VOCAB[tok] == "EOF":
                break
            chaine.append(VOCAB[tok])
        # garantie de terminaison TYPÉE : si on s'arrête hors de 'champ' (max_len
        # atteint sans EOF), on complète par l'opérateur qui ramène au type 'champ'.
        while _type_apres(chaine) != "champ" and len(chaine) < max_len + 3:
            for op, (te, ts) in OPERATEURS.items():
                if te == _type_apres(chaine) and ts == "champ":
                    chaine.append(op)
                    break
            else:
                break
        return chaine

    def echantillonner(self, objectif, max_len=4):
        """Émet un programme en ÉCHANTILLONNANT (exploration), sous masque de type.
        Retourne (chaine, log_prob) où log_prob est un tenseur DIFFÉRENTIABLE = somme
        des log π des tokens choisis (EOF compris) — la quantité que REINFORCE remonte."""
        h = self.emb_obj(torch.tensor([objectif], device=DEVICE)).squeeze(0)
        tok = torch.tensor(BOS, device=DEVICE)
        chaine = []
        log_prob = torch.zeros((), device=DEVICE)
        entropie = torch.zeros((), device=DEVICE)
        for _ in range(max_len):
            h = self.gru(self.emb_tok(tok).unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            logits = self._logits(h)
            masque = _masque_type(chaine).to(DEVICE)
            logits = logits.masked_fill(~masque, float("-inf"))
            dist = torch.distributions.Categorical(logits=logits)
            tok = dist.sample()
            log_prob = log_prob + dist.log_prob(tok)
            entropie = entropie + dist.entropy()      # garde l'exploration vivante
            if VOCAB[tok] == "EOF":
                break
            chaine.append(VOCAB[tok])
        return chaine, log_prob, entropie

    def perte_imitation(self, objectif, cible_chaine):
        """Entropie croisée (teacher forcing) pour reproduire `cible_chaine`."""
        h = self.emb_obj(torch.tensor([objectif], device=DEVICE)).squeeze(0)
        tok = torch.tensor(BOS, device=DEVICE)
        cibles = [VOCAB.index(op) for op in cible_chaine] + [VOCAB.index("EOF")]
        perte = torch.zeros((), device=DEVICE)
        chaine = []
        for c in cibles:
            h = self.gru(self.emb_tok(tok).unsqueeze(0), h.unsqueeze(0)).squeeze(0)
            logits = self._logits(h)
            masque = _masque_type(chaine).to(DEVICE)
            logits = logits.masked_fill(~masque, float("-inf"))
            perte = perte + torch.nn.functional.cross_entropy(
                logits.unsqueeze(0), torch.tensor([c], device=DEVICE))
            tok = torch.tensor(c, device=DEVICE)
            if VOCAB[c] != "EOF":
                chaine.append(VOCAB[c])
        return perte / len(cibles)


def entrainer_par_imitation(mode_b, exemples, pas=400, lr=5e-3):
    """exemples : liste de (objectif_idx, chaine_gagnante_de_mode_A)."""
    opt = torch.optim.Adam(mode_b.parameters(), lr=lr)
    for _ in range(pas):
        for obj, chaine in exemples:
            opt.zero_grad()
            perte = mode_b.perte_imitation(obj, chaine)
            perte.backward()
            opt.step()
    log("mode_b", "imitation_terminee", n_exemples=len(exemples), pas=pas)


def entrainer_par_renforcement(mode_b, recompense, objectifs, pas=400, lr=3e-3,
                               beta_baseline=0.1, beta_entropie=0.05):
    """REINFORCE (§31.6, après l'imitation ou À LA PLACE) : Mode B ÉCHANTILLONNE un
    programme, on le mesure par la récompense R = G − λ·coût (`recompense(obj, chaine)`),
    et on remonte ∇ log π · (R − baseline). Aucun professeur : l'orchestrateur DÉCOUVRE
    le meilleur programme par la seule récompense (Principe 2 : appris par renforcement
    selon le contexte). La baseline (moyenne mobile de R par objectif) réduit la variance.

    Régularisation par ENTROPIE avec RECUIT (`beta_entropie`, décru linéairement) : sans
    elle, la politique partagée s'effondre tôt sur le programme le plus facile à
    échantillonner (le plus court) et n'explore plus les programmes plus longs mais
    meilleurs pour d'autres objectifs — l'avantage positif n'arrive alors jamais. On
    maintient l'exploration au début, puis on laisse exploiter en fin.

    `objectifs` : itérable d'indices d'objectif (0..n-1). `recompense(obj, chaine)→float`."""
    objectifs = list(objectifs)
    opt = torch.optim.Adam(mode_b.parameters(), lr=lr)
    base = {o: 0.0 for o in objectifs}
    for t in range(pas):
        beta_ent = beta_entropie * (1 - t / pas)      # recuit : exploration → exploitation
        for obj in objectifs:
            chaine, log_prob, entropie = mode_b.echantillonner(obj)
            r = float(recompense(obj, chaine))
            avantage = r - base[obj]
            base[obj] = (1 - beta_baseline) * base[obj] + beta_baseline * r
            opt.zero_grad()
            (-avantage * log_prob - beta_ent * entropie).backward()
            opt.step()
    log("mode_b", "renforcement_termine", n_objectifs=len(objectifs), pas=pas)
