"""Les PULSIONS qui meuvent l'agent (étape 17, §4 de la conception).

Un seul objectif DOMINANT gouverne l'action à la fois (argmax + hystérésis, §15.3) —
jamais un pot-commun pondéré. On réutilise `memoires.TableBesoins` (faim + ennui/temps
perdu + le Schmitt-trigger d'hystérésis déjà prouvé) et `decision_action.reflexe_cable`
(la douleur est un garde-fou câblé PRIORITAIRE, jamais mélangé aux besoins). On ajoute les
pulsions COGNITIVES lues sur l'état des modules (curiosite.py) :

    curiosité      = incertitude prédictive haute  → aller vers l'inconnu (découvrir)
    apprentissage  = progrès d'apprentissage > 0    → « jouer » : progresser sur une tâche entamée
    bullage        = un module MAÎTRISÉ couvre la situation → agir sans effort, libérer du calcul

La récompense `recompense()` (gain sucre − douleur + progrès − pénalité de temps) alimente
la planification (A*, étape 19) et le renforcement nocturne (étape 20). Aucun terme n'est
câblé à la géométrie du monde : faim = compteur interne, douleur = signal, etc.
"""
from .config import CONFIG
from .decision_action import reflexe_cable
from .logger import log
from .memoires import TableBesoins

COGNITIVES = ("curiosite", "apprentissage", "bullage")


class Pulsions:
    def __init__(self):
        self.besoins = TableBesoins()             # faim + ennui (= temps perdu) + hystérésis
        self.douleur = 0.0
        for k in COGNITIVES:
            self.besoins.etats[k] = 0.0

    def maj(self, evenements=(), vitesse_norme=0.0,
            curiosite=0.0, apprentissage=0.0, bullage=0.0):
        """Met à jour les pulsions. `evenements` : liste du monde {"sucre","baton"}.
        Les 3 pulsions cognitives sont fournies (calculées sur l'état des modules)."""
        evenements = list(evenements)
        self.besoins.mettre_a_jour(evenements, vitesse_norme)         # faim, ennui
        n_baton = sum(1 for e in evenements if e == "baton")
        # douleur : décroissance SOUSTRACTIVE (persiste ~1/decroissance pas) + choc
        self.douleur = min(1.0, max(0.0, self.douleur - CONFIG["decroissance_douleur"])
                           + CONFIG["douleur_baton"] * n_baton)
        self.besoins.etats["curiosite"] = float(curiosite)
        self.besoins.etats["apprentissage"] = float(apprentissage)
        self.besoins.etats["bullage"] = float(bullage)

    def objectif_dominant(self):
        """Réflexe douleur PRIORITAIRE (court-circuite tout) ; sinon le besoin dominant
        par argmax + hystérésis sur les 5 pulsions restantes (faim, ennui, curiosité,
        apprentissage, bullage)."""
        if reflexe_cable(self.douleur) is not None:
            return "douleur"
        return self.besoins.besoin_dominant()

    def recompense(self, evenements=(), progres=0.0):
        """r_t pour planification/renforcement : gain sucre − douleur + progrès − temps.
        La pénalité de temps est faible (dominée par tout vrai gain) mais décisive quand
        rien d'autre ne bouge — anti-tergiversation."""
        evenements = list(evenements)
        r = CONFIG["recompense_sucre"] * sum(1 for e in evenements if e == "sucre")
        r -= CONFIG["douleur_baton"] * sum(1 for e in evenements if e == "baton")
        r += float(progres)
        r -= CONFIG["penalite_temps"]
        return float(r)

    def etat(self):
        """Instantané des pulsions (instrumentation / viewer)."""
        d = {k: round(float(v), 4) for k, v in self.besoins.etats.items()}
        d["douleur"] = round(float(self.douleur), 4)
        return d
