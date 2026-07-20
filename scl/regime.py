"""Détection de régime en ESPACE-CHAMP (corrige le goulot de STATUS §5bis).

La composition des étapes 6-9 travaillait sur le latent compressé OPAQUE du
module 1 (prédiction ~57 %). Ici les modules-régime prédisent champ(T-1) → champ(T)
directement (transitions conv déjà prouvées à 84 %, étape 2a). Avantages :

- le résidu est **le rappel objets**, naturellement dans [0,1] : SANS UNITÉ, sans
  normalisation ni ratio au prior (règle l'enseignement #4 par construction) ;
- un vent transverse fait bouger les objets en Y : la prédiction (décalage en X) les
  rate → le rappel chute → nouveauté immédiatement visible (ce qui échouait en latent).

Un module-régime = un `ModuleAutoencodeur` entraîné en mode transition sur UN régime,
puis VERROUILLÉ (§1.4). Naissance sur surprise CONFIRMÉE (le meilleur rappel reste bas)
+ grâce. Le nouveau-né garde la main le temps d'apprendre (sinon oubli — enseignement #6).
L'état par module (rappel lissé, maturité, verrou, étalon) est tenu par le détecteur.
"""
from collections import deque

from .config import CONFIG
from .logger import log, log_verbeux
from .module_ae import ModuleAutoencodeur


class _EtatModule:
    __slots__ = ("module", "n_maj", "ema_rappel", "verrouille", "reference")

    def __init__(self, module):
        self.module = module
        self.n_maj = 0
        self.ema_rappel = None
        self.verrouille = False
        self.reference = None


class DetecteurRegimeChamp:
    def __init__(self):
        self.regimes = {}          # id → _EtatModule
        self.champ_prec = None
        self._ema_meilleur = None  # rappel lissé du meilleur module (décision)
        self._hist_meilleur = deque(maxlen=2 * CONFIG["fenetre_progres_regime"])
        self._grace = 0
        self._actif = None

    def _progresse(self):
        """Le meilleur module PROGRESSE-t-il encore ? (§28.1 : ne pas créer tant que
        G monte — un module qui apprend n'est pas un échec confirmé). Compare la
        moyenne de rappel récente à la précédente."""
        w = CONFIG["fenetre_progres_regime"]
        if len(self._hist_meilleur) < 2 * w:
            return True                       # pas assez de recul → prudence : on attend
        recent = sum(list(self._hist_meilleur)[-w:]) / w
        ancien = sum(list(self._hist_meilleur)[-2 * w:-w]) / w
        return recent > ancien + CONFIG["epsilon_progres_regime"]

    # ------------------------------------------------------------- rappels / N2
    def rappels(self, champ_prec, champ):
        """Rappel de prédiction de CHAQUE module sur la transition (0..1)."""
        return {mid: e.module.fidelite_transition(champ_prec, champ)["rappel"]
                for mid, e in self.regimes.items()}

    def identifier(self, champ):
        """N2 (§29.2) + familiarité (§29.1), SANS apprendre : quel module explique
        le mieux la transition, et à quel point (rappel du meilleur ∈ [0,1])."""
        actif, rap, familiarite = None, {}, 0.0
        if self.champ_prec is not None and self.regimes:
            rap = self.rappels(self.champ_prec, champ)
            actif = max(rap, key=rap.get)
            familiarite = rap[actif]
        self.champ_prec = champ
        return actif, rap, familiarite

    # ------------------------------------------------------------- naissance
    def _naissance(self):
        mid = f"regime_{len(self.regimes)}"
        self.regimes[mid] = _EtatModule(ModuleAutoencodeur(mid))
        # un module conv champ→champ est LENT à devenir compétent (~2000 pas, cf.
        # étape 2a) : sa grâce doit couvrir cet apprentissage, sinon on en crée un
        # autre avant qu'il ait fini (sur-création mesurée).
        self._grace = CONFIG["grace_regime"]
        self._ema_meilleur = None
        self._hist_meilleur.clear()             # le nouveau-né repart d'une trace vierge
        self._actif = mid
        log("regime", "naissance_module_regime", module=mid, n_modules=len(self.regimes))
        return mid

    def etape(self, champ, apprendre=True):
        """Détecte le régime, entraîne le module responsable, fait NAÎTRE un module
        si aucun n'explique la transition (surprise confirmée + grâce)."""
        detecte, rap = None, {}
        if self.champ_prec is not None:
            if not self.regimes:
                detecte = self._naissance()
            else:
                rap = self.rappels(self.champ_prec, champ)
                detecte = max(rap, key=rap.get)
                a = CONFIG["ema_residu_composition"]
                self._ema_meilleur = rap[detecte] if self._ema_meilleur is None else \
                    a * self._ema_meilleur + (1 - a) * rap[detecte]
                self._hist_meilleur.append(rap[detecte])
                if self._grace > 0:
                    self._grace -= 1
                    detecte = self._actif           # le nouveau-né garde la main
                elif self._ema_meilleur < CONFIG["seuil_rappel_inexplique"] and not self._progresse():
                    # inexpliqué ET plus de progrès (plateau bas) → vraie surprise
                    detecte = self._naissance()
                else:
                    self._actif = detecte
            if apprendre and detecte is not None:
                self._apprendre(detecte, self.champ_prec, champ)
            log_verbeux("regime", "detection", detecte=detecte,
                        rappels={k: round(v, 2) for k, v in rap.items()})
        self.champ_prec = champ
        return detecte, rap

    def _apprendre(self, mid, champ_prec, champ):
        e = self.regimes[mid]
        if e.verrouille:
            return
        e.module.entrainer_transition(champ_prec, champ)
        e.n_maj += 1
        r = e.module.fidelite_transition(champ_prec, champ)["rappel"]
        a = CONFIG["ema_residu_composition"]
        e.ema_rappel = r if e.ema_rappel is None else a * e.ema_rappel + (1 - a) * r
        # verrouillage asymétrique (§1.4) : compétent et stable → figé, étalon mémorisé
        if e.n_maj >= CONFIG["maturite_module_vitesse"] and e.ema_rappel > CONFIG["seuil_rappel_maitrise"]:
            e.verrouille = True
            e.reference = e.ema_rappel
            log(mid, "verrouillage_module_regime", rappel_reference=round(e.ema_rappel, 3))

    def verrouiller_tout(self):
        for e in self.regimes.values():
            if e.ema_rappel is not None:
                e.verrouille = True
                e.reference = e.reference or e.ema_rappel
