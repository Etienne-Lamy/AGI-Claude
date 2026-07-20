"""ORCHESTRATEUR — Mode A « dirigé » (§31.5) : recherche typée de programmes.

Un programme compose des OPÉRATEURS TYPÉS pour transformer un signal de départ en
une cible ancrée (§31.1 : la cible est du vérifiable — ici le vrai champ suivant).
Le typage rend la plupart des programmes absurdes IMPOSSIBLES à énumérer (§31.3),
ce qui réduit l'espace. On entraîne chaque programme terminal, on mesure son GAIN
DE PRÉDICTIBILITÉ `G` contre le prior trivial, et on garde le meilleur au sens
`valeur = G − λ·coût` (§31.4-5). Aucune préférence câblée : l'orchestrateur
DÉCOUVRE par la mesure quelle composition prédit le mieux.

Opérateurs (type_entrée → type_sortie), tous adossés à des modules déjà prouvés :
    compresser      champ  → latent      (module 1, goulot ; coût = dim latent)
    generer         latent → champ       (générateur du même module)
    predire_champ   champ  → champ        (transition conv, étape 2a)
    predire_latent  latent → latent       (prédicteur dense, étape 2b)
"""
import numpy as np

from .config import CONFIG
from .logger import log
from .module_ae import ModuleAutoencodeur, PredicteurAbstrait

OPERATEURS = {
    "compresser": ("champ", "latent"),
    "generer": ("latent", "champ"),
    "predire_champ": ("champ", "champ"),
    "predire_latent": ("latent", "latent"),
}


# ---------------------------------------------------------------- énumération typée
def enumerer_programmes(profondeur_max=3, type_depart="champ", type_cible="champ"):
    """Tous les enchaînements d'opérateurs bien typés `type_depart`→…→`type_cible`,
    longueur ≤ profondeur_max. Le typage élague : on ne peut pas `generer` un champ
    ni `predire_latent` un champ, etc."""
    progs = []

    def dfs(type_courant, chaine):
        if type_courant == type_cible and chaine:
            progs.append(list(chaine))
        if len(chaine) >= profondeur_max:
            return
        for op, (te, ts) in OPERATEURS.items():
            if te == type_courant:
                chaine.append(op)
                dfs(ts, chaine)
                chaine.pop()

    dfs(type_depart, [])
    # dédoublonne et retire les programmes sans aucune prédiction (pure
    # reconstruction : ils prédiraient l'entrée, pas la cible — mais on les GARDE
    # comme témoins que la mesure de G doit rejeter).
    uniques = []
    for p in progs:
        if p not in uniques:
            uniques.append(p)
    return uniques


# ------------------------------------------------------------------ exécution d'un programme
class Programme:
    """Compose des modules selon une chaîne d'opérateurs, s'entraîne, prédit."""

    def __init__(self, chaine, dim_latent=None):
        self.chaine = chaine
        self.dim_latent = dim_latent or CONFIG["dim_latent_vision"]
        self.compresseur = None     # partagé par compresser/generer
        self.pred_champ = None
        self.pred_latent = None
        if "compresser" in chaine or "generer" in chaine:
            self.compresseur = ModuleAutoencodeur("prog_comp", dim_latent=self.dim_latent)
        if "predire_champ" in chaine:
            self.pred_champ = ModuleAutoencodeur("prog_pred_champ")
        if "predire_latent" in chaine:
            self.pred_latent = PredicteurAbstrait("prog_pred_latent", dim_latent=self.dim_latent)

    def cout(self):
        """Coût MDL approché (§31.4) : somme des goulots des modules mobilisés."""
        c = 0
        if self.compresseur is not None:
            c += self.dim_latent
        if self.pred_champ is not None:
            c += self.pred_champ.dim_latent
        if self.pred_latent is not None:
            c += self.dim_latent
        return c

    def entrainer(self, champ_prec, champ):
        """Entraîne chaque module sur SON signal (local, §0). Le compresseur apprend
        la reconstruction ; le prédicteur-champ la transition ; le prédicteur-latent
        la transition dans l'espace compressé."""
        if self.compresseur is not None:
            self.compresseur.entrainer(champ)                       # reconstruction
        if self.pred_champ is not None:
            self.pred_champ.entrainer_transition(champ_prec, champ)  # transition champ
        if self.pred_latent is not None and self.compresseur is not None:
            z_prec = self.compresseur.encoder(champ_prec).detach()
            z = self.compresseur.encoder(champ).detach()
            self.pred_latent.entrainer(z_prec, z)                    # transition latent

    def predire(self, champ_prec):
        """Exécute la chaîne (§31.2) : threade le signal à travers les opérateurs."""
        import torch
        signal = champ_prec
        for op in self.chaine:
            if op == "compresser":
                signal = self.compresseur.encoder(signal).detach()
            elif op == "generer":
                signal = self.compresseur.generer(signal)
            elif op == "predire_champ":
                signal = self.pred_champ.predire(signal)             # champ prédit
            elif op == "predire_latent":
                signal = self.pred_latent.predire(signal)
        return signal


# ------------------------------------------------------------------ Mode A (choix par valeur)
def _rappel(champ_pred, champ_vrai):
    import torch
    from .module_ae import DEVICE
    p = torch.as_tensor(champ_pred, dtype=torch.float32, device=DEVICE).reshape(-1)
    v = torch.as_tensor(champ_vrai, dtype=torch.float32, device=DEVICE).reshape(-1)
    obj = v > CONFIG["seuil_objet_vision"]
    n = int(obj.sum())
    if not n:
        return 1.0
    return int((((p - v).abs() < 0.2) & obj).sum()) / n


def evaluer_programme(prog, paires_eval):
    """Rappel moyen de prédiction, et G vs le prior trivial « champ inchangé »."""
    r, rt = [], []
    for a, b in paires_eval:
        r.append(_rappel(prog.predire(a), b))
        rt.append(_rappel(a, b))                     # prior : prédire a pour b
    rm, rtm = float(np.mean(r)), float(np.mean(rt))
    g = (rm - rtm) / (1.0 - rtm) if rtm < 1.0 else 0.0
    return rm, rtm, g


def mode_A(flux_train, paires_eval, profondeur_max=3, pas=1500, lam=None):
    """Énumère les programmes typés, entraîne chacun, mesure G, renvoie le classement
    par valeur = G − λ·coût_normalisé. C'est l'A* de profondeur bornée de §31.5
    (ici sans heuristique d'élagage : l'espace typé est petit)."""
    lam = lam if lam is not None else CONFIG["lambda_cout_programme"]
    resultats = []
    for chaine in enumerer_programmes(profondeur_max):
        prog = Programme(chaine)
        for champ_prec, champ in flux_train(pas):
            if champ_prec is not None:
                prog.entrainer(champ_prec, champ)
        rm, rtm, g = evaluer_programme(prog, paires_eval)
        cout = prog.cout()
        valeur = g - lam * cout / 100.0
        resultats.append({"chaine": chaine, "rappel": round(rm, 3), "trivial": round(rtm, 3),
                          "G": round(g, 3), "cout": cout, "valeur": round(valeur, 3)})
        log("orchestrateur", "programme_evalue", chaine=chaine, G=round(g, 3),
            cout=cout, valeur=round(valeur, 3))
    resultats.sort(key=lambda r: -r["valeur"])
    return resultats
