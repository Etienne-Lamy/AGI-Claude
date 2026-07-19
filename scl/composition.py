"""Composition de modules — détecter la vitesse par une CHAÎNE de modules, pas un
réseau monolithique. Briques génériques composables (Architecture §27) :

  • ModuleDelai  : recopie l'output d'un module au pas T-1 (registre/trace). Empilable
                   pour T-2, T-3…
  • ModuleVitesse: transforme un latent T-1 → latent T (prédiction). Un module par
                   RÉGIME de transformation ; il ÉMERGE quand un régime nouveau
                   (vitesse inédite) apparaît et que les modules existants échouent.

Composition évaluée à DEUX niveaux (comme demandé) :
    champ(T) → [compresseur] → z(T)
    z(T-1) --[délai]--> [module vitesse] --> ẑ(T)
      (1) résidu LATENT   : ‖ẑ(T) − z(T)‖   (compare à l'output réel du compresseur)
      (2) résidu CHAMP    : [générateur du compresseur](ẑ(T)) vs champ(T) réel
Le module-vitesse de plus faible résidu = régime (vitesse) détecté. Un résidu
partout élevé = surprise → naissance d'un nouveau module (émergence).

Chaque module produit un latent à HAUTE VALEUR ; on peut insérer des modules à
différents niveaux et continuer à composer avec l'aval (le générateur).
"""
from collections import deque

import torch

from .config import CONFIG
from .logger import log, log_verbeux
from .module_ae import DEVICE


class ModuleDelai:
    """Registre : .sortie = valeur poussée au pas précédent (T-1)."""

    def __init__(self):
        self._buf = None

    @property
    def sortie(self):
        return self._buf

    def pousser(self, v):
        self._buf = None if v is None else v.detach()


class ModuleVitesse:
    """latent(T-1) → latent(T). MLP. Un régime de transformation (une vitesse)."""

    def __init__(self, id, dim, dim_cachee=128):
        self.id = id
        self.dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim_cachee), torch.nn.ReLU(),
            torch.nn.Linear(dim_cachee, dim_cachee), torch.nn.ReLU(),
            torch.nn.Linear(dim_cachee, dim)).to(DEVICE)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=CONFIG["lr_vision_ae"])
        self.erreurs = deque(maxlen=CONFIG["fenetre_incertitude"])
        self.n_maj = 0
        self.verrouille = False       # verrouillage asymétrique (§1.4)
        self._ema_rel = None          # résidu RELATIF lissé (compétence vs prior trivial)
        self._ema_abs = None          # erreur ABSOLUE lissée (latent normalisé)
        self.reference = None         # erreur absolue au verrou = étalon de reconnaissance

    def predire(self, z):
        with torch.no_grad():
            return self.net(z.reshape(1, self.dim)).squeeze(0)

    def entrainer(self, z_prec, z_present, res_rel=None):
        """VERROUILLÉ = n'apprend plus. Sans ce verrou, un module compétent se
        ré-entraîne sur le régime suivant et OUBLIE le sien (mesuré) : plus aucune
        spécialisation, donc plus aucune détection. Le verrou est ce qui force la
        NAISSANCE d'un module pour un régime nouveau (§1.4, §4.5)."""
        if self.verrouille:
            return None
        pred = self.net(z_prec.reshape(1, self.dim)).squeeze(0)
        perte = torch.mean((pred - z_present) ** 2)
        self.opt.zero_grad(); perte.backward(); self.opt.step()
        e = float(perte.detach()); self.erreurs.append(e)
        self.n_maj += 1
        if res_rel is not None:
            a = CONFIG["ema_residu_composition"]
            self._ema_rel = res_rel if self._ema_rel is None else a * self._ema_rel + (1 - a) * res_rel
        self._ema_abs = e if self._ema_abs is None else a * self._ema_abs + (1 - a) * e
        if (self.n_maj >= CONFIG["maturite_module_vitesse"] and self._ema_rel is not None
                and self._ema_rel < CONFIG["seuil_maturite_vitesse"]):
            self.verrouille = True
            self.reference = self._ema_abs       # étalon de RECONNAISSANCE (§29.1)
            log(self.id, "verrouillage_module_vitesse", n_maj=self.n_maj,
                residu_relatif=round(self._ema_rel, 3), reference=round(self._ema_abs, 4))
        return e

    def incertitude(self):
        return sum(self.erreurs) / len(self.erreurs) if self.erreurs else float("inf")


def _residu_champ(compresseur, z_pred, champ):
    """(2) régénère un champ depuis le latent prédit et le compare au réel :
    1 − rappel objets (0 = parfait)."""
    with torch.no_grad():
        champ_pred = compresseur.generer(z_pred)                      # (t*t,) valeurs
    cible = torch.as_tensor(champ, dtype=torch.float32, device=DEVICE).reshape(-1)
    obj = cible > CONFIG["seuil_objet_vision"]
    n = int(obj.sum())
    if not n:
        return 0.0
    juste = ((champ_pred - cible).abs() < 0.2) & obj
    return 1.0 - int(juste.sum()) / n


class DetecteurVitesse:
    """Compose compresseur + délai + modules-vitesse ; détecte le régime et fait
    ÉMERGER un module quand un régime nouveau apparaît."""

    def __init__(self, compresseur, seuil_surprise=None):
        self.comp = compresseur
        self.dim = compresseur.dim_latent
        self.delai = ModuleDelai()
        self.vitesses = {}
        self.seuil = seuil_surprise if seuil_surprise is not None else CONFIG["seuil_surprise_composition"]
        # §4.5 : on ne crée JAMAIS sur un incident isolé — il faut une surprise
        # CONFIRMÉE (n pas consécutifs) ; et après une naissance, un délai de
        # grâce laisse le nouveau-né apprendre avant de rejuger (sinon on crée un
        # module à chaque pas, le nouveau-né étant lui aussi non entraîné).
        self.grace = CONFIG["grace_creation_composition"]
        self._ema_res = None          # résidu lissé du meilleur module (décision)
        self._grace_restante = 0
        self._actif = None            # module qui apprend actuellement (garde la main en grâce)
        # statistiques courantes du latent : les modules travaillent sur un latent
        # NORMALISÉ (le latent brut du compresseur a des magnitudes arbitraires —
        # sans ça les résidus valent des dizaines et tout paraît surprenant).
        self._moy = None
        self._var = None

    def _maj_stats(self, z):
        if self._moy is None:
            self._moy = z.clone(); self._var = torch.ones_like(z)
        else:
            a = CONFIG["ema_stats_latent"]
            self._moy = a * self._moy + (1 - a) * z
            self._var = a * self._var + (1 - a) * (z - self._moy) ** 2

    def _norm(self, z):
        return (z - self._moy) / (self._var.sqrt() + 1e-6)

    def _denorm(self, zn):
        return zn * (self._var.sqrt() + 1e-6) + self._moy

    def _naissance(self):
        vid = f"vitesse_{len(self.vitesses)}"
        self.vitesses[vid] = ModuleVitesse(vid, self.dim)
        self._ema_res = None          # le nouveau-né n'est pas jugé sur le passé
        self._grace_restante = self.grace
        self._actif = vid
        log("composition", "naissance_module_vitesse", module=vid,
            n_modules=len(self.vitesses))
        return vid

    def _residus(self, z_prec, z, champ):
        """Résidus (latent relatif borné, champ) de CHAQUE module sur la transition."""
        base = float(torch.mean((z_prec - z) ** 2)) + 1e-6
        out = {}
        for vid, mv in self.vitesses.items():
            z_pred = mv.predire(z_prec)
            res_lat = min(float(torch.mean((z_pred - z) ** 2)) / base,
                          CONFIG["plafond_residu_composition"])
            res_abs = float(torch.mean((z_pred - z) ** 2))     # erreur ABSOLUE (latent normalisé)
            out[vid] = (round(res_lat, 3),
                        round(_residu_champ(self.comp, self._denorm(z_pred), champ), 3),
                        round(res_abs, 4))
        return out

    def _inexplique(self, residus):
        """(module le mieux placé, score d'INEXPLICATION ≥ 0 ; > 1 = régime non expliqué).

        UNE SEULE mesure gouverne la reconnaissance ET la décision de créer. Dès que
        des spécialistes verrouillés existent, on compare l'erreur ABSOLUE à l'ÉTALON
        du module (son erreur quand il était compétent) : c'est insensible à
        l'amplitude du changement. Le résidu relatif au prior trivial ne sert que
        tant qu'aucun étalon n'existe — sinon un monde qui défile vite rend le prior
        mauvais et masque la nouveauté (mesuré : aucune naissance sous vent transverse)."""
        avec_ref = [v for v in residus if self.vitesses[v].reference]
        if avec_ref:
            ratio = lambda v: residus[v][2] / (self.vitesses[v].reference + 1e-9)
            vid = min(avec_ref, key=ratio)
            return vid, ratio(vid) / CONFIG["seuil_ratio_inexplique"]
        vid = min(residus, key=lambda v: residus[v][0])
        return vid, residus[vid][0] / self.seuil

    def identifier(self, champ):
        """**N2** : quel module explique le mieux la transition courante — SANS
        apprendre ni créer. C'est le signal de régime (discret, très faible
        cardinalité) sur lequel se construit le niveau supérieur (§29.2), et la
        mesure de familiarité connu/inconnu (§29.1).
        Retourne (id_actif|None, residus, familiarite) où familiarite = 1 − résidu
        du meilleur (gain vs le prior trivial)."""
        z_brut = self.comp.encoder(champ).detach()
        self._maj_stats(z_brut)
        z = self._norm(z_brut)
        z_prec = self.delai.sortie
        actif, residus, familiarite = None, {}, 0.0
        if z_prec is not None and self.vitesses:
            residus = self._residus(z_prec, z, champ)
            # RECONNAISSANCE : erreur ABSOLUE (le latent est déjà normalisé, donc sans
            # unité) rapportée à l'ÉTALON du module — son erreur quand il a été jugé
            # compétent. SURTOUT PAS le résidu relatif au prior trivial : celui-ci est
            # confondu par l'amplitude du changement — un monde qui défile plus vite
            # rend le prior trivial mauvais et fait paraître TOUT module meilleur
            # (mesuré : la familiarité MONTAIT quand le vent se levait).
            def _ratio(vid):
                ref = self.vitesses[vid].reference
                return residus[vid][2] / (ref + 1e-9) if ref else float("inf")
            actif = min(residus, key=_ratio)
            r = _ratio(actif)
            familiarite = 1.0 / max(1.0, r) if r != float("inf") else 0.0
        self.delai.pousser(z)
        return actif, residus, familiarite

    def etape(self, champ, apprendre=True):
        z_brut = self.comp.encoder(champ).detach()                    # z(T) latent (compresseur gelé)
        self._maj_stats(z_brut)
        z = self._norm(z_brut)                                        # latent NORMALISÉ
        z_prec = self.delai.sortie                                    # z(T-1) normalisé
        detecte, residus = None, {}
        if z_prec is not None:
            residus = self._residus(z_prec, z, champ)   # (relatif borné, champ, absolu)
            if not self.vitesses:
                detecte = self._naissance()          # tout premier régime
            else:
                detecte, meilleur = self._inexplique(residus)
                # décision sur le résidu LISSÉ (EMA) du meilleur module : robuste au
                # bruit pas-à-pas, et c'est déjà une « surprise confirmée » (§4.5) —
                # un compteur de pas consécutifs ne déclenchait jamais (queue lourde).
                a = CONFIG["ema_residu_composition"]
                self._ema_res = meilleur if self._ema_res is None else a * self._ema_res + (1 - a) * meilleur
                if self._grace_restante > 0:
                    # le nouveau-né GARDE LA MAIN le temps d'apprendre son régime :
                    # sinon l'argmin réélit un ancien module, qui se fait écraser
                    # (oubli) au lieu que chacun se spécialise.
                    self._grace_restante -= 1
                    detecte = self._actif
                elif self._ema_res > 1.0:            # score normalisé : >1 = inexpliqué
                    detecte = self._naissance()      # régime inexpliqué → émergence
                else:
                    self._actif = detecte
            if apprendre:
                self.vitesses[detecte].entrainer(z_prec, z, residus.get(detecte, (None,))[0])
            log_verbeux("composition", "detection", detecte=detecte, residus=residus)
        self.delai.pousser(z)
        return detecte, residus
