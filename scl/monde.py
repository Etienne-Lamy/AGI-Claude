"""Monde simulé SCL (spécifié par les prompts 10-11).

- Grille 2D INFINIE, génération procédurale déterministe par chunks.
- L'agent ne se téléporte pas : il accélère. Accélérations permises :
  (0,0), (±1,0), (0,±1). Vitesse par composante bornée à ±v_max (2).
- Perception : champ visuel niveaux de gris 10×10 centré sur l'agent,
  historique de 3 frames. Le corps (petit cercle) est visible au centre.
- Objets : sucres (récompense) et bâtons (douleur), déposés au hasard.
- Proprioception : vitesse (vx, vy) et dernière accélération (ax, ay).
"""
import numpy as np

from .config import CONFIG
from .logger import log, log_verbeux

VAL_VIDE = 0.0
VAL_BATON = 0.5
VAL_SUCRE = 1.0
VAL_CORPS = 0.25

ACCELERATIONS_PERMISES = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


class Monde:
    def __init__(self, graine=None):
        self.graine = graine if graine is not None else CONFIG["graine_monde"]
        self.taille_chunk = CONFIG["taille_chunk"]
        self.taille_perception = CONFIG["taille_perception"]
        self.n_frames = CONFIG["n_frames"]
        self.v_max = CONFIG["v_max"]
        self._chunks = {}
        self.consommes = set()
        self.agent_pos = np.array([0, 0], dtype=np.int64)
        self.vitesse = np.array([0, 0], dtype=np.int64)
        self.derniere_accel = np.array([0, 0], dtype=np.int64)
        self.vent = np.array([0, 0], dtype=np.int64)   # déplacement subi (voir appliquer_action)
        self.historique_vision = [self._frame() for _ in range(self.n_frames)]
        self.compteurs = {"sucre": 0, "baton": 0, "steps": 0}
        log("monde", "creation", graine=self.graine)

    # ------------------------------------------------- génération procédurale
    def _objets_chunk(self, cx, cy):
        key = (cx, cy)
        if key not in self._chunks:
            rng = np.random.default_rng(
                abs(hash((cx, cy, self.graine))) % (2 ** 31))
            objets = {}
            n = self.taille_chunk
            for dx in range(n):
                for dy in range(n):
                    r = rng.random()
                    x, y = cx * n + dx, cy * n + dy
                    if abs(x) <= 1 and abs(y) <= 1:
                        continue  # zone de naissance dégagée
                    if r < CONFIG["densite_sucre"]:
                        objets[(x, y)] = "sucre"
                    elif r < CONFIG["densite_sucre"] + CONFIG["densite_baton"]:
                        objets[(x, y)] = "baton"
            self._chunks[key] = objets
        return self._chunks[key]

    def objet_en(self, x, y):
        if (x, y) in self.consommes:
            return None
        n = self.taille_chunk
        return self._objets_chunk(x // n, y // n).get((x, y))

    # ------------------------------------------------------------- dynamique
    def appliquer_action(self, accel):
        """accel : couple parmi ACCELERATIONS_PERMISES. Retourne les événements."""
        accel = tuple(int(a) for a in accel)
        if accel not in ACCELERATIONS_PERMISES:
            # projection sur l'accélération permise la plus proche
            accel = min(ACCELERATIONS_PERMISES,
                        key=lambda a: (a[0] - accel[0]) ** 2 + (a[1] - accel[1]) ** 2)
        self.derniere_accel = np.array(accel, dtype=np.int64)
        self.vitesse = np.clip(self.vitesse + self.derniere_accel,
                               -self.v_max, self.v_max)
        # Le VENT (défaut : nul) est un déplacement subi, qui s'ajoute à la vitesse
        # propre SANS la modifier : l'agent garde le même état interne mais le monde
        # défile autrement — c'est un RÉGIME DE DYNAMIQUE nouveau, pas une perception
        # nouvelle. Sert à tester la localisation d'échec (§29.4).
        deplacement = self.vitesse + self.vent
        ancienne = self.agent_pos.copy()
        self.agent_pos = self.agent_pos + deplacement
        evenements = []
        # collisions sur les cellules traversées (trajectoire discrète simple)
        n_pas = int(max(abs(deplacement[0]), abs(deplacement[1]), 1))
        for i in range(1, n_pas + 1):
            p = ancienne + (deplacement * i) // n_pas
            obj = self.objet_en(int(p[0]), int(p[1]))
            if obj:
                self.consommes.add((int(p[0]), int(p[1])))
                self.compteurs[obj] += 1
                evenements.append(obj)
        self.compteurs["steps"] += 1
        if evenements:   # les collisions sont toujours journalisées
            log("monde", "action_appliquee", accel=accel,
                vitesse=self.vitesse.tolist(),
                position=self.agent_pos.tolist(), evenements=evenements)
        else:
            log_verbeux("monde", "action_appliquee", accel=accel,
                        vitesse=self.vitesse.tolist(),
                        position=self.agent_pos.tolist(), evenements=[])
        return evenements

    # ------------------------------------------------------------ perception
    def _frame(self):
        t = self.taille_perception
        frame = np.zeros((t, t), dtype=np.float32)
        x0 = int(self.agent_pos[0]) - t // 2
        y0 = int(self.agent_pos[1]) - t // 2
        for i in range(t):
            for j in range(t):
                obj = self.objet_en(x0 + i, y0 + j)
                if obj == "sucre":
                    frame[i, j] = VAL_SUCRE
                elif obj == "baton":
                    frame[i, j] = VAL_BATON
        frame[t // 2, t // 2] = VAL_CORPS  # le corps, visible dans le champ
        return frame

    def percevoir(self):
        frame = self._frame()
        self.historique_vision.append(frame)
        self.historique_vision = self.historique_vision[-self.n_frames:]
        contexte = {
            "vision": np.stack(self.historique_vision),      # (3, 10, 10)
            "proprio": np.array([self.vitesse[0], self.vitesse[1],
                                 self.derniere_accel[0], self.derniere_accel[1]],
                                dtype=np.float32),
            "position": self.agent_pos.copy(),
        }
        log_verbeux("monde", "perception", position=self.agent_pos.tolist(),
                    n_sucres_visibles=int((frame == VAL_SUCRE).sum()),
                    n_batons_visibles=int((frame == VAL_BATON).sum()))
        return contexte

    def objets_visibles(self):
        """Positions relatives (di, dj) des objets dans le champ courant."""
        t = self.taille_perception
        frame = self.historique_vision[-1]
        sucres, batons = [], []
        for i in range(t):
            for j in range(t):
                d = (i - t // 2, j - t // 2)
                if frame[i, j] == VAL_SUCRE:
                    sucres.append(d)
                elif frame[i, j] == VAL_BATON:
                    batons.append(d)
        return sucres, batons
