# SCL — État des lieux (document de REPRISE À FROID)
> Auteur : Etienne Lamy
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

Tests : `python3 -m pytest tests/ -q` → **238 passed** (~2 min 30, plusieurs tests GPU).
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
| 8 | **Horizons T+h + branches** | `etape8_horizons.py` | `python3 -m scl.etape8_horizons --horizon 8` | G(h)=+17/+28/+26/+18/+6/+1/+4/−3 % → **horizon naturel T+5** (mesuré, pas choisi) ; branches correctes (saturation à v_max) |
| 9 | **Vent : localiser l'échec** ⚠ partiel | `etape9_vent.py` | `python3 -m scl.etape9_vent --vent 0 2` | signature §29.4 obtenue (N1 **intact**, N2 **effondrée**) mais **variable** et **aucune naissance** — corrigé à l'étape 10 |
| 10 | **Détection de régime en ESPACE-CHAMP** (lève §5bis) | `regime.py` | `python3 -m scl.etape10_regime_champ` | détection nette (56-59 % on-régime vs 20-27 % ailleurs), 3/3, et **VENT transverse détecté** (familiarité 59→28 %, module né) — ce que le latent opaque ne faisait jamais |
| 11 | **Orchestrateur Mode A** (A\* typé, choix par valeur) | `orchestrateur.py` | `python3 -m scl.etape11_orchestrateur` | classe 7 programmes typés par G−λ·coût et **CHOISIT `predire_champ` (G=64 %)**, rejette la chaîne latent opaque (36 %) et la reconstruction pure (2 %) → **redécouvre §5bis seul** |
| 12 | **Boucle JOUR→NUIT** (capturer / comprendre) | `memoire_episodique.py`, `nuit.py` | `python3 -m scl.etape12_jour_nuit` | vent capturé le jour (5 épisodes cohérents, familiarité ~25 %), **tous COMPRIS la nuit** (rejeu prédit > seuil) |
| 13 | **Orchestrateur Mode B** (LLM+attention, distillation de A) | `mode_b.py` | `python3 -m scl.etape13_mode_b` | émetteur typé autorégressif, apprend par imitation ; émet le bon programme PAR OBJECTIF sans recherche (voir §7bis) |
| 14 | **Mode B par RENFORCEMENT** (découverte sans professeur) | `mode_b.py` | `python3 -m scl.etape14_reinforce` | depuis R=G−λ·coût seule, init aléatoire : **découvre les 2 optima** (2/2) grâce à l'entropie recuite ; objectifs opposés dans la table de récompense (voir §7bis) |
| 15 | **Auto-réglage §28.4 branché** (réversible, sur un vrai levier) | `autoreglage.py` | `python3 -m scl.etape15_autoreglage --pas_regime 3000` | règle `grace_regime` par la SEULE mesure : 1000→**9 modules** (obs −0.09), 2000→5 (obs +0.29), **4000→3 = l'idéal** (obs +0.44, gardé) → **corrige la sur-création** (résidu étapes 10/6) sans valeur donnée à la main |
| 16 | **Conséquence d'une ACTION sur le champ prévisible** (phase action) | `action.py` | `python3 -m scl.etape16_action_champ` | un module champ→champ par action (copie d'efférence) ; effet SOUTENU (vitesse saturée) → matrice **diagonale 57 % vs 31 %** (+26 %). Leçon : l'effet d'UN pas (vitesse remise à 0) est trop faible/proche (matrice plate ~50 %) → mesurer l'effet soutenu, ce que la navigation exploite |
| 17 | **Pulsions & objectif dominant** (faim/douleur/curiosité/apprentissage/bullage/temps) | `pulsions.py` | `python3 -m scl.etape17_pulsions` | réutilise `TableBesoins` (argmax+hystérésis) + `reflexe_cable` ; un seul objectif dominant, douleur prioritaire. Mesuré : cognitif **39 %→2 %** hors-douleur (vision se maîtrise → curiosité saturée), hystérésis **2 %** de bascules. Constat : agent aléatoire **76 % en douleur** → motive la PLANIFICATION (18-19) |
| 18 | **Boucle MPC : le g() appris pilote l'action** | `planification.py` | `python3 -m scl.etape18_boucle_action` | `ModeleRecompense` r̂(champ,action) (mémoire de rejeu+mini-lot, ench. #10) ; glouton ε → **évite la douleur** : bâtons 37.8 (hasard) → **29.0** (planifié fin), **−23 %**, décroissant. Sucre lointain ≈ hasard (le glouton 1-pas ne vise pas loin → A* étape 19 + crédit nocturne étape 20) |
| 19 | **Valeur Q=g+h par TD** (mécanisme OK, ⚠ négatif en ligne) | `planification.py` | `python3 -m scl.etape19_valeur` | `ModeleValeurQ` Q(champ,action) par TD (bootstrap `r+γ maxQ`) ; test unitaire : le crédit se propage vers la bonne action. Mais **en ligne, la visée du sucre NE craque PAS** (gain +1.8/1000 = bruit) : récompense trop rare/non façonnée → il faut CONCENTRER l'apprentissage sur les rares épisodes récompensés (étape 20). Résultat négatif **honnête**, attendu |
| 20 | **Rejeu nocturne AMONT** (mécanisme OK, ⚠ visée non émergée) | `nuit_action.py` | `python3 -m scl.etape20_nuit_action` | retours **n-pas** priorisant les épisodes sucre ; **test unitaire : le crédit d'une récompense finale REMONTE 5 pas en amont** (bug n-pas trouvé/corrigé : la récompense terminale était perdue). Mais éval **gelée sur monde frais** : politique **dégénérée en vol rectiligne** (0 bâton, 0 sucre) → la visée du sucre **n'émerge pas sur CHAMP BRUT**. Prescription : planifier sur l'**état compact objet** (slot-attention étape 4) — frontière suivante |

> **Phase Action (16→20) — état honnête.** Toute l'ossature est bâtie et unit-testée :
> modèle de transition action-conditionné (16), pulsions/objectif dominant (17), g() appris
> (18), valeur Q=g+h par TD (19), rejeu nocturne à crédit amont (20). **Acquis mesurés** :
> l'action est prévisible (16, diag. 57 %), l'évitement de la douleur ÉMERGE (18, −23 % vs
> hasard), le crédit remonte n pas (20, test unitaire). **Non acquis, dit franchement** : la
> **visée du sucre lointain n'émerge pas** — sur champ brut, la récompense (contact rare) est
> dominée par la pénalité de temps et le greedy dégénère en vol rectiligne. Ce n'est pas un
> échec des mécanismes (tous validés en isolation) mais de la **représentation d'état** : il
> faut planifier sur l'**état compact objet** (positions issues de la slot-attention, étape 4),
> pas sur les 100 pixels bruts. **C'est LA tâche de la prochaine session.**

| 21 | **Curiosité anti-vol-rectiligne** (seedé) | `etape21_curiosite.py` | `python3 -m scl.etape21_curiosite` | politique = action la MOINS bien prévue (incertitude max du modèle de transition). Curieuse : entropie **0.40**, **5/5 actions maîtrisées** « de proche en proche ». Exploitante (argmin) : entropie **0**, se fige sur 1 action, **1/5 maîtrisée** → la curiosité sort du figement |
| 22 | **Placement maternel → VISÉE DU SUCRE ÉMERGE** ✅ (seedé) | `maternage.py`, `etape22_maternage.py` | `python3 -m scl.etape22_maternage` | la « maman » pose l'agent à 2-3 cases d'un sucre et DÉMONTRE l'action vers lui (récompense dense) → Q apprend « sucre visible → aller vers lui ». Éval **gelée, monde FRAIS, sans placement** : sucre **58.4 vs 51.2 hasard (+14 %)** — **la visée du sucre GÉNÉRALISE** (là où le champ-brut seul échouait, étape 20). Contrepartie honnête : **+16 bâtons** (fonce vers les objets, moins prudent — arbitrage sucre/douleur à affiner) |

> **Horizons — jusqu'où la prévision monte (réponse à une question de l'auteur).**
> Prévoir T+h ne crée PAS un module par horizon : c'est **exécuter la même règle** N3 h fois
> (§29.3). De nouveaux modules ne naissent que sur un **régime/phénomène nouveau** (surprise),
> pas par horizon. Le niveau atteint dépend de CE qu'on prévoit : le **régime** (vitesse, variable
> LENTE, attracteur à saturation) reste prévisible **loin** — mesuré `G(h)` = +27..+31 % jusqu'à
> **T+10** (etape8 `--horizon 10 --pas_regime 1500 --pas_action 3500`) ; le **champ détaillé**
> décroche vite (erreurs de décalage cumulées → ~T+1-2 utile). Un prédicteur d'horizon est **jeté
> quand G(h)→0** (il ne bat plus « rien ne change ») : c'est le garde-fou MDL qui borne la pile.
> ⚠ Le pipeline régime→N3 est **fragile** (un run peut dégénérer : N3→identité, G(h)=0 partout) —
> à stabiliser.

### Phase UPGRADE (étapes 23→28) — plan `SCL_plan_upgrade.md` : raisonner en OBJETS

| # | Livrable | Harnais | Résultat mesuré |
|---|---|---|---|
| 23 | **Perception OBJET** (fondation) ✅ | `python3 -m scl.etape23_perception_objet` | champ → `{(catégorie,position)}` (VQ émergent, catégories pures) ; prédiction T+1 = **décaler les positions par la vitesse**. Rappel **région prévisible = 100 % à toutes les vitesses** (vs 57-84 % en pixels !) ; global 90 % (le résidu = objets entrant par le bord, **plafond info-théorique** du capteur 10×10). Compression **5.4 objets vs 100 pixels** |
| 24 | **Action = ACCÉLÉRATION + compositionnalité** ✅ | `python3 -m scl.etape24_action_objet` | transition `(E,v,a)→(decaler(E,clip(v+a)),v')`. **`(2,0)=(1,0)∘(1,0)` à 100 %** (la vitesse 2 se simule par double usage du décalage de 1 — aucun module (2,0) dédié) ; multi-pas prévisible **T+1..T+6 = 100→90 %** (vrai déroulé, plus de `(1,0)×10` figé) ; branches (1,0)×4 vs (0,1)×4 = **68 %** de divergence objets → futurs distincts |

Visualisation (viewer **v7**) — 2 commandes :
```bash
python3 -m scl.demo_viewer --log demo.jsonl          # produit le log (modules + orchestrateur)
python3 viewer.py --log demo.jsonl --port 8400        # 2e terminal → http://localhost:8400
```
Trois panneaux : **champ VU vs PRÉVU** (zoom réglable, déborde du cadre au-delà d'un seuil),
**modules** (naissance + courbe de rappel par module, verrouillage), **graphe de branchement**
de l'orchestrateur (nœuds champ/latent, arêtes = opérateurs, programme choisi surligné,
programmes classés par valeur). Vocabulaire de log `viewer` : meta/phase/champ/modules_etat/
programme_choisi + `regime/naissance_module_regime` + `orchestrateur/programme_evalue`.

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
11. **Un critère « relatif au prior trivial » est confondu par l'amplitude du
    changement** : un monde qui défile plus vite rend le prior mauvais et fait paraître
    TOUT module meilleur (mesuré : la familiarité MONTAIT quand le vent se levait).
    → pour la RECONNAISSANCE, erreur **absolue** (latent normalisé) rapportée à
    l'**étalon du module** ; le relatif ne sert qu'à « bat-il le trivial ? ».
12. **Vérifier qu'un stimulus « nouveau » l'est vraiment** : un vent (1,0) sur une
    vitesse (1,0) donne un déplacement (2,0) — un régime DÉJÀ appris. Le système
    avait raison de le reconnaître ; c'était l'expérience qui était mal conçue.
13. **La qualité d'un niveau plafonne le niveau au-dessus** (étape 7) : N3 apprend
    bien la règle action→régime (+31 %), et ses erreurs se localisent **au niveau du
    dessous** — le vocabulaire N2 confond v=(1,0) et (2,0) dans un même module et
    garde un module parasite non associé. Illustration directe de §29.4 : diagnostiquer
    le niveau le **plus bas** anormal avant de toucher au niveau supérieur.

---

## 5bis. LE GOULOT D'ÉTRANGLEMENT ACTUEL (à traiter en priorité)

Toute la hiérarchie (étapes 6→9) est bâtie sur le **latent compressé OPAQUE** du
module 1. Or les mesures montrent que c'est le maillon faible :

| représentation | qualité de prédiction mesurée |
|---|---|
| latent compressé opaque (64) — **utilisé par la hiérarchie** | **57 %** (chaîne 1→2→1) |
| latent spatial (non compressant) | 80 % |
| **représentation OBJET** (slot attention, étape 4) | **94 % reconstruction / 80 % prédiction triviale** |

Conséquence en cascade : si les modules-vitesse prédisent mal, un changement de
régime ne dégrade que **faiblement** leur erreur → la nouveauté devient
difficilement détectable (c'est pourquoi le vent ne déclenche pas de naissance),
et le vocabulaire N2 confond des régimes voisins (v=(1,0) et (2,0)), ce qui
plafonne N3.

**Action prioritaire pour la suite** : rebâtir la composition sur la
**représentation objet** (`module_attention.liste_objets`) au lieu du latent
opaque. Un régime devient alors « comment les objets se déplacent », un vent
transverse est immédiatement visible (les objets partent en Y), et la prédiction
redevient triviale (décaler les positions). C'est aussi ce qui lève la dette
§27.4 (tête 4-classes donnée → catégories émergentes).

---

## 7bis. Orchestrateur (§31) — ce qui est CONSTRUIT

L'orchestrateur du cahier des charges §31 est bâti et validé sur ses briques
principales (Mode A, Mode B, mémoire, nuit). Deux MODES, comme prévu :

- **Mode A « dirigé »** (`orchestrateur.py`) : énumère les programmes bien TYPÉS
  (le typage élague), entraîne chacun, mesure `G`, classe par `valeur = G − λ·coût`.
  Sans préférence câblée, il **redécouvre §5bis** (champ-direct > latent opaque) et
  rejette reconstruction pure et programmes redondants (étape 11).
- **Mode B « appris »** (`mode_b.py`) : émetteur de programmes AUTORÉGRESSIF à
  émission TYPÉE (masque dur sur le vocabulaire), conditionné par l'OBJECTIF. Deux
  voies de §31.6, toutes deux validées :
  - IMITATION de Mode A (étape 13) : le meilleur programme DIFFÈRE selon l'objectif
    (prédire → `predire_champ` ; reconstruire → `compresser→generer`) et Mode B émet
    le bon **sans refaire la recherche** — il amortit Mode A.
  - RENFORCEMENT sans professeur (étape 14) : depuis la seule récompense
    `R = G − λ·coût`, init aléatoire, Mode B **découvre** les deux optima (2/2). Il
    fallait une **entropie recuite** : sinon la politique partagée s'effondre sur le
    programme le plus court (`predire_champ`, local-optimum honnête de la reconstruction
    à R=+0.16) et n'échantillonne jamais le vrai gagnant (R=+0.63). Les objectifs
    s'OPPOSENT dans la table de récompense → dépendance au contexte réelle, apprise.
- **Temps réel + mémoire** (`memoire_episodique.py`) : on ne retient que le
  non-régénérable (graine = champ initial + actions + modules actifs + résidu) ; on
  ne capture que le surprenant ; hystérésis obligatoire pour ne pas fragmenter.
- **Nuit** (`nuit.py`) : rejeu d'un épisode + apprentissage dédié jusqu'à le prédire ;
  critère de « COMPRIS » mesurable. Étape 12 : imprévu capturé le jour, compris la nuit.
- **Auto-réglage §28.4** (`autoreglage.py`) : boucle RÉVERSIBLE générique — ne garde un
  changement d'hyperparamètre que s'il améliore un observable, sinon revert (asymétrie).
  Branchée en vrai sur `grace_regime` contre le résidu de sur-création (étape 15).

**NON encore fait de §31** (honnête) : la **bascule mesurée A↔B** (quand cesser de
chercher pour émettre) ; l'**instrumentation dashboard §31.10** des nouveaux objets
(programmes, épisodes, réglages). Résidus : rappels conv modérés (~44-59 %, modules
sous-entraînés dans les runs courts). La **sur-création est résolue** par l'auto-régleur
(étape 15 : `grace_regime=4000` → 3 modules = l'idéal à `pas=3000`).

> Insight étape 15 : la grâce optimale DÉPEND de l'horizon d'entraînement (1000 à
> `pas=1000`, 4000 à `pas=3000`) — une grâce en **pas absolus** est intrinsèquement
> liée à la durée. Dette : le critère de naissance gagnerait à compter en **progrès**
> (déjà via `_progresse`) plutôt qu'en pas fixes, ou à indexer la grâce sur l'horizon.

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

0. **PRIORITÉ — rebâtir la hiérarchie sur la représentation OBJET** (§5bis) : c'est
   le goulot qui plafonne N2, N3, la détection de nouveauté et les horizons.
1. **Familiarité `F` par niveau** — fait (`DetecteurVitesse.identifier`) : erreur
   absolue rapportée à l'étalon du module, en écarts-types de sa propre erreur
   (auto-calibré). À réévaluer une fois la base objet en place.
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
