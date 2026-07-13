"""Logger d'audit SCL.

Exigence expérimentale : CHAQUE action de CHAQUE module est journalisée
(création, activation, entraînement, verrouillage, rupture, découpe, rêve...).
Format : une ligne JSON par action (JSONL), plus un tampon mémoire consultable
par les tests. Champs standard : n (compteur), jour, step, acteur, action.
"""
import json
import threading

_TRONQUE = 40  # taille max des vecteurs sérialisés dans le log


def _propre(v):
    """Sérialise proprement tenseurs / arrays / sets pour JSON."""
    try:
        import torch
        if isinstance(v, torch.Tensor):
            v = v.detach().flatten().tolist()
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            v = v.flatten().tolist()
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
    except ImportError:
        pass
    if isinstance(v, (set, frozenset, tuple)):
        v = list(v)
    if isinstance(v, list):
        v = [_propre(x) for x in v[:_TRONQUE]] + (["..."] if len(v) > _TRONQUE else [])
    if isinstance(v, float):
        if v != v or v == float("inf") or v == float("-inf"):
            return None   # NaN/inf : invalides en JSON standard (tue le viewer)
        v = round(v, 6)
    if isinstance(v, dict):
        v = {str(k): _propre(x) for k, x in v.items()}
    return v


class AuditLogger:
    def __init__(self, chemin=None, console=False, max_memoire=20000,
                 verbeux=False):
        self.chemin = chemin
        self.console = console
        self.verbeux = verbeux   # False : les forwards/gates ne sont pas écrits
        self.f = open(chemin, "a", encoding="utf-8") if chemin else None
        self.jour = 0
        self.step = 0
        self.n = 0
        self.records = []           # tampon mémoire (pour tests / inspection)
        self.max_memoire = max_memoire
        self._lock = threading.Lock()

    def set_temps(self, jour=None, step=None):
        if jour is not None:
            self.jour = jour
        if step is not None:
            self.step = step

    def log(self, acteur, action, **details):
        with self._lock:
            rec = {"n": self.n, "jour": self.jour, "step": self.step,
                   "acteur": str(acteur), "action": str(action)}
            for k, v in details.items():
                rec[k] = _propre(v)
            self.n += 1
            self.records.append(rec)
            if len(self.records) > self.max_memoire:
                del self.records[: self.max_memoire // 2]
            if self.f:
                self.f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                if self.n % 20 == 0:   # flush fréquent : suivi temps réel
                    self.f.flush()
            if self.console:
                print(f"[{rec['jour']}:{rec['step']}] {acteur} :: {action} :: "
                      + " ".join(f"{k}={v}" for k, v in details.items() if not isinstance(v, (list, dict))))

    def filtrer(self, acteur=None, action=None):
        return [r for r in self.records
                if (acteur is None or r["acteur"] == acteur)
                and (action is None or r["action"] == action)]

    def fermer(self):
        if self.f:
            self.f.flush()
            self.f.close()
            self.f = None


_GLOBAL = AuditLogger()


def configurer(chemin=None, console=False, verbeux=False):
    global _GLOBAL
    _GLOBAL.fermer()
    _GLOBAL = AuditLogger(chemin, console, verbeux=verbeux)
    return _GLOBAL


def obtenir():
    return _GLOBAL


def est_verbeux():
    return _GLOBAL.verbeux


def log(acteur, action, **details):
    _GLOBAL.log(acteur, action, **details)


def log_verbeux(acteur, action, **details):
    """Journalisation fine (forwards, gates, buffers) — uniquement si le
    logger est en mode verbeux. L'observabilité structurelle (entraînements,
    condensateurs, ruptures, nuits) reste toujours journalisée."""
    if _GLOBAL.verbeux:
        _GLOBAL.log(acteur, action, **details)


def set_temps(jour=None, step=None):
    _GLOBAL.set_temps(jour, step)
