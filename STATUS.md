# SCL — État des lieux (document de REPRISE À FROID)

> Ce fichier est fait pour qu'une conversation NEUVE soit productive immédiatement,
> sans que l'auteur ait à réexpliquer le projet. Lire ceci en entier d'abord.
> Docs de fond (source de vérité, lues à la demande) : `SCL - Vision et Strategie.md`,
> `SCL_fondements_mathematiques.md`, `Architecture SCL Code v2.md` (dont **§27-§30**,
> écrits pour l'orchestrateur), `README v2.md`.

---

## 1. Le projet en cinq lignes

SCL = un « cerveau » qui **construit sa propre structure** : des **modules
détecteur/générateur** créés dynamiquement, et un **orchestrateur** qui les
**compose**. Le monde 2D (grille, sucres/bâtons, agent qui accélère) n'est qu'un
**bac à sable de PREUVE D'ÉMERGENCE** — l'objectif final est d'étendre les principes
à de la vision/audition complexe de robots. Donc : **le moins de codage en dur
possible**, et toute adaptation manuelle doit être **documentée comme dette** ou
comme outil à enseigner à l'orchestrateur.

---

## 2. Principes non négociables (l'auteur a dû me les rappeler — ne pas les reperdre)

1. **Parcimonie / MDL est le moteur** (§5) : un module doit **RÉDUIRE ses E/S**
   (sortie < entrée). La taille **interne** peut être grosse (générateur de qualité) ;
   seul le **goulot** compte. Le calcul doit porter sur de l'abstraction à valeur,
   pas sur du signal brut.
2. **Le choix d'architecture EST une action de l'orchestrateur**, pas un réglage que
   je fixe : catalogue de formes → essayer → garder par MDL. Naïf d'abord, **appris
   par renforcement** ensuite selon le contexte.
3. **Les catégories doivent ÉMERGER** (aucune classe donnée). Le cerveau classifie
   d'abord ; chaque catégorie devient un module qui l'**identifie et la régénère**.
4. **L'orchestrateur compose, ne calcule jamais lui-même** (§10.2). Les modules sont
   des **compositions de FONCTIONS**, pas des combinaisons linéaires. Planifier le
   futur ressemble à **du code** (d'où l'orchestrateur pensé comme un LLM à attention).
5. **Verrouillage asymétrique** (§1.4) : plancher jamais plafond. **Sans verrou, un
   module compétent est écrasé par le régime suivant** → aucune spécialisation
   (vérifié empiriquement, cf. §5.6 ci-dessous).
6. **Créer sur surprise CONFIRMÉE**, jamais sur un incident isolé (§4.5), avec un
   délai de grâce.
7. Apprentissage **local** (aucun gradient entre modules), cycle **jour/nuit**,
   **activation creuse**.
8. Le monde est un bac à sable : on **teste l'émergence**, on ne maximise pas un score
   de tâche. (Ex. : « manger des sucres » n'est PAS l'objectif.)

---

## 3. Environnement (obligatoire avant tout run)

```bash
cd ~/IA-Ubuntu/AGI-Claude
source ~/venv_pytorch_kepler/bin/activate && source ~/pytorch-kepler/dist/setup_env.sh
```
GPU : GTX Titan Black (Kepler, sm_35), torch 1.12 compilé maison. **Watchdog** : ne
pas empiler plusieurs entraînements GPU simultanés (« launch timed out ») — entraîner
**un module à la fois**. Le GPU récupère seul après un timeout.

Tests : `python3 -m pytest tests/ -q` → **211 passed** (~1 min 30, plusieurs tests GPU).
Git : dépôt `github.com:Etienne-Lamy/AGI-Claude` (SSH). **Ne jamais pousser sans
accord explicite de l'auteur.**

---

## 4. Ce qui est CONSTRUIT et VALIDÉ (étapes 1→7)

Chaque étape a un harnais reproductible. Les chiffres sont mesurés, pas estimés.

| # | Capacité | Fichier | Harnais | Résultat mesuré |
|---|---|---|---|---|
| 1 | **Compresser & reconstruire** le champ | `module_ae.py` | `python3 -m scl.etape1_vision --pas 4000` | goulot **64 < 100**, F1 **90 %** |
| 2a | **Prédire T-1→T**, fiabilité = **indicateur de vitesse** | `module_ae.entrainer_transition` | `python3 -m scl.etape2_prediction --pas 2500` | rappel **84 %** à la vitesse entraînée vs **~20 %** ailleurs |
| 2b | Chaîne module1→module2→module1 | `composition.py` | `python3 -m scl.etape2b_chaine --vitesse 1 1 --log etape2.jsonl` | 80 % (latent spatial) / 57 % (latent compressé opaque) |
| 3 | **Orchestrateur naïf** : taille choisie par **MDL** | `orchestrateur_naif.py` | `python3 -m scl.etape3_catalogue --pas 2000` | sur [8..96] choisit **dim 48** (ni trop petit ni trop grand) |
| 4 | **Attention/masquage** → objets → **prédiction triviale** | `module_attention.py` | `python3 -m scl.etape4_attention --pas 8000 --log etape4.jsonl` | reconstruction **94 %** ; prédire en **décalant les objets** (aucun réseau) : **80 %** |
| 5 | **Classification ÉMERGENTE** (aucune étiquette) | `classification_emergente.py` | `python3 -m scl.etape5_classification --pas 4000` | **4 catégories** émergent (sur 6, reste élagué), **100 % pures**, reconstruction **100 %** |
| 6 | **Composition qui DÉTECTE la vitesse** | `composition.py` | `python3 -m scl.etape6_composition --pas_regime 1500` | 4 modules nés, **3/3 régimes couverts**, les 2 niveaux concordent |
| 7 | **Hiérarchie N2→N3** : action → changement de régime | `hierarchie.py` | `python3 -m scl.etape7_hierarchie` | exactitude **57 %** vs trivial **38 %** → **gain +31 %** ; règle lisible et physiquement correcte sur les régimes bien séparés |

Visualisation : `python3 viewer.py --log <fichier>.jsonl --port 8400` → http://localhost:8400
(panneau VU vs PRÉVU en carrés, incertitude, événements d'émergence).

---

## 5. Enseignements empiriques (chèrement acquis — voir Architecture §28.3)

Chaque échec ci-dessous a un **signal objectif** qui le trahit et un **correctif
générique** : c'est la matière de l'auto-réglage futur de l'orchestrateur.

1. **Effondrement à zéro** : sur un champ ~90 % vide, une MSE se minimise en sortant
   du noir. → perte **pondérée** / classification ; **ne jamais conclure sur une
   perte scalaire**, suivre le rappel de la classe rare.
2. **Plafond insensible à la capacité** : un MLP plafonne à ~44 % que le latent fasse
   48, 100 ou 200. → ce n'est pas la taille, c'est le **biais inductif** (il faut de
   la **convolution** avant le goulot). Faire un **balayage de capacité** comme diagnostic.
3. **Compression ≠ fidélité pure** : conv pleine résolution reconstruit à 100 % mais
   **EXPANSE** (300 > 100). Le goulot doit réduire.
4. **Critères sans unité** : le latent brut a des magnitudes arbitraires (résidus à
   50-60, seuils absurdes). → **normaliser**, et exprimer tout critère en **ratio au
   prior trivial** « rien ne change ».
5. **Statistiques à queue lourde** : moyenne ≫ médiane ; un compteur de pas consécutifs
   ne déclenche jamais. → **borner + lisser (EMA)** avant de décider.
6. **Verrouillage = condition de la spécialisation** : sans lui, un module compétent
   se ré-entraîne sur le régime suivant et **oublie** le sien → 2/3 ; avec lui → **3/3**.
7. **Création qui s'auto-déclenche** : 799 modules en 800 pas. → surprise **confirmée**
   + **grâce** (le nouveau-né doit apprendre avant d'être rejugé).
8. **Slots naïfs ne se spécialisent pas** (21 %) : il faut une **compétition explicite**
   (softmax sur les slots) + itération → 94 %.
9. **Champ réceptif = portée sémantique** : des features à contexte donnent des
   catégories sales ; un encodeur **1×1** (apparence locale) donne des catégories pures.
10. **Batch-1 en ligne ne converge pas** sur entrées quasi-aléatoires → **mémoire de
    rejeu + mini-lot** (rester en ligne, gagner en stabilité).
11. **La qualité d'un niveau plafonne le niveau au-dessus** (étape 7) : N3 apprend
    bien la règle action→régime (+31 %), et ses erreurs se localisent **au niveau du
    dessous** — le vocabulaire N2 confond v=(1,0) et (2,0) dans un même module et
    garde un module parasite non associé. Illustration directe de §29.4 : diagnostiquer
    le niveau le **plus bas** anormal avant de toucher au niveau supérieur.

---

## 6. Dettes assumées (codage en dur à retirer)

- `module_attention.py` reconstruit via une **tête 4-classes DONNÉE** (`VALEURS`) :
  à basculer sur les **catégories émergentes** de `classification_emergente.py`
  (Architecture §27.4).
- `curiosite.py` / `dynamique.py` (action pilotée par la curiosité) : construits
  avant le recadrage « étape par étape ». **Ne font pas partie du chemin validé** —
  à réévaluer ou retirer une fois l'orchestrateur en place.
- Le monde ne correspond pas encore tout à fait à la spec de l'auteur : accélérations
  **diagonales** (9 au lieu de 5) et **monde fini avec murs** (vitesse annulée au mur)
  restent à implémenter ; aujourd'hui monde infini procédural, 5 accélérations.

---

## 7. Feuille de route (Architecture §29.6)

1. **Familiarité `F` par niveau** — connu vs inconnu, et *quel* régime : `F = max_m G_m`
   (gain de prédictibilité), argmax = régime identifié. Quasi gratuit, tout s'appuie dessus.
2. **Profil de résidus par niveau** — le point de branchement est le niveau le **plus bas**
   anormal ; puis carte de résidu intra-niveau (l'attention sert de loupe).
3. **N2 → N3** (fait, étape 7) : constance de vitesse, puis **accélération = transition
   de régime conditionnée par l'action**.
4. **Horizons T+2…T+n** : étendre tant que `G(h) > 0` ; la courbe `G(h)` donne l'horizon naturel.
5. **Mémoire épisodique par résidu** : ne mémoriser que le **non-régénérable**
   (graine = latent initial + actions + identité des modules actifs + résidu),
   critère de suffisance = *le rejeu reproduit-il l'épisode ?* Alimente le cycle nocturne.
6. Puis l'**orchestrateur** qui choisit/compose lui-même (RL sur le journal
   `(contexte, correctif, ΔG)` — Architecture §28.4).

Test d'émergence prévu par l'auteur, plus tard : introduire du **« vent »** ponctuel
qui perturbe la vitesse → doit faire naître un module dédié (N1 intact, N2 s'effondre).

---

## 8. Historique condensé

- **2026-07-17** — bascule Windows/Cowork → Claude Code sous WSL ; CUDA validé (matmul
  4.3 ms GPU vs 10.5 s CPU) ; audit théorie↔code.
- **2026-07-18 (jour)** — POC « navigation » : l'agent mangeait (192 sucres, 15.7
  pas/sucre) **mais par câblage** (coordonnées vérité-terrain + distance géométrique).
  **Rejeté** : trahissait la thèse.
- **2026-07-18 (soir)** — refonte « curiosité » (`curiosite.py`, `dynamique.py`) :
  action pilotée par l'incertitude. Écarté ensuite car ne validait pas les briques une à une.
- **2026-07-18 (nuit)** — **découverte critique** : l'ancienne vision (JEPA masqué)
  s'effondrait à zéro (0 cellule-objet reconstruite) — « vision maîtrisée » mesurait du vent.
- **2026-07-19** — reprise **étape par étape**, une étape validée avant la suivante :
  étapes 1→7 ci-dessus, puis rédaction d'Architecture **§27** (contraintes + boîte à
  outils), **§28** (auto-diagnostic/auto-réglage), **§29** (hiérarchie, horizons,
  localisation d'échec, mémoire), **§30** (planification = composition de fonctions).
