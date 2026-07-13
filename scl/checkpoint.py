"""Persistance de l'état entraîné — état complet (§6) : graphe, monde,
besoins, registres, accumulateurs, mémoire de travail, simulateurs (portés
par le graphe, §10.2). Générique : sauvegarde/charge un dict nommé de
composants plutôt qu'une liste figée de champs — `boucle.EtatSCL` transporte
tout le reste.

Attention : un checkpoint est lié à la version du code (pickle). Après une
modification structurelle du code, repartir d'un run frais.
"""
import os
import pickle

from .logger import log


def sauvegarder(chemin, **composants):
    tmp = chemin + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(composants, f)
    os.replace(tmp, chemin)   # écriture atomique
    log("checkpoint", "sauvegarde", chemin=chemin, composants=list(composants.keys()))


def charger(chemin):
    with open(chemin, "rb") as f:
        composants = pickle.load(f)
    log("checkpoint", "chargement", chemin=chemin, composants=list(composants.keys()))
    return composants


def existe(chemin):
    return bool(chemin) and os.path.exists(chemin)
