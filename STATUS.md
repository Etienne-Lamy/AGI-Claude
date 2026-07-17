# État des lieux — POC SCL (2026-07-03)

## Mise à jour 2026-07-18 — POC fonctionnel : navigation apprise

Session autonome. **Jalon atteint : l'agent mange, gère sa vitesse, évite les bâtons, et APPREND son corps pour naviguer seul.**

**Diagnostic initial (harnais `scl/eval_poc.py`, nouveau)** — la baseline ne mangeait presque rien (2.3 sucres / 3000 pas, `sucres==batons`, effondrement total après le jour 0). Deux pathologies :
- *Deadlock du réflexe de douleur* : une fois v=0 le réflexe renvoyait (0,0) et re-freinait sans fin → agent paralysé ~60 pas pendant que la douleur décroît. Corrigé (`inne.reflexe_frein` rend la main à v=0).
- *Navigation sans gestion de vitesse* : l'heuristique notait l'accélération par alignement en ignorant la vitesse → ressort non amorti qui survole le sucre sans jamais atterrir (oscillation en boucle limite). C'était exactement le cœur du problème "gérer la vitesse".

**Corrections + résultats** (commits `6490481`, puis modèle appris) :
1. **Navigation par rollout horizon-1** (`boucle._scores_actions`) : chaque accélération évaluée sur la position PRÉDITE (v'=clip(v+accel)) + pénalité de franchissement de bâton. → sucres 2.3→**192**, bâtons 2.3→**0.7**, steps/sucre 1750→**15.7**, soutenu sur 10 jours.
2. **Modèle du corps APPRIS** (`scl/prevision.py`, divergence #1) : MLP (v,accel)→v' entraîné en ligne, auto-supervisé (cible = vitesse réellement observée). Quand sa fiabilité π dépasse le seuil, la navigation bascule de l'instinct (vérité-terrain) au modèle appris (§15.1 transfert par confiance). → le modèle atteint err≈1e-5, π≈1.0, et **pilote ~60-65% des pas** en mangeant autant (180 vs 192). L'agent a appris sa propre dynamique et navigue avec.
3. **Crash NaN corrigé** (`attention.py`) : les pointeurs (REINFORCE) divergeaient sur longues séries → logits inf/nan → crash `multinomial`. Ajout : clip de gradient + softmax/tirage robustes (repli uniforme). Indispensable pour les runs multi-rounds.

**Workflow 2 commandes** (stabilisé) : `run_poc.py` a maintenant `--checkpoint cerveau.pkl` par défaut (reprise auto multi-rounds) et un bilan enrichi (steps/sucre, fiabilité du corps, % navigation apprise). Viewer inchangé. Reprise multi-rounds vérifiée : le modèle appris et les compteurs persistent et continuent de progresser d'un round au suivant.

**Validation longue durée** (diagnostic 20 jours × 300 pas, graine 1) : alimentation soutenue ~18 sucres/jour (361 au total), **modules stables à 6 — aucune prolifération** (le SPRT de création fait son travail : un seul module créé puis plus rien), **aucun NaN** sur 6000 pas (le correctif orchestrateur tient), temps stable ~9 s/jour (pas de ralentissement superlinéaire). La fiabilité du modèle du corps oscille (0.6–1.0) avec repli gracieux sur l'instinct lors des creux — comportement voulu (§15.1), non bloquant.

**Reste / prochaines frontières** :
- **Jeu pendant l'ennui (le "rêve")** : bloqué en amont, découverte de cette session — `faim` sature à 1.0 (l'agent n'arrive pas à se rassasier au rythme actuel), donc `ennui` (plafonné à 0.5) ne devient JAMAIS le besoin dominant. Le jeu est un phénomène post-satiété : il faut d'abord équilibrer densité de sucre / vitesse de faim pour que l'agent puisse s'ennuyer. **Décision de game-design à trancher par toi** (`densite_sucre`, dynamique de `faim`) avant que le mécanisme de jeu ait un sens. L'infrastructure (besoin ennui, hystérésis, scores d'action ennui) est déjà en place et n'attend que ça.
- Divergence #4 (allocation WFQ `allocation_attention.allouer_capacite`) et `consolidation_n_vers_un` (§9, seule fonction réellement manquante) — ne se déclencheraient pas sur ce POC à 1 module créé ; complétude théorique, faible priorité.
- **Environnement appris** : le module visuel CNN tourne mais son rôle dans la navigation reste indirect (la navigation s'appuie sur les positions relatives perçues + le modèle du corps appris, pas encore sur une prévision APPRISE du champ visuel — piste : forward model visuel §1.3 "vitesse→image", qui adresserait aussi l'échec historique du module vision décrit dans README v2).

## Mise à jour 2026-07-17 — reprise sous Claude Code

- Bascule de Claude Cowork (Windows, non exécutant) vers Claude Code sous WSL Ubuntu, répertoire de travail = ce dossier. Dépôt git connecté à `github.com:Etienne-Lamy/AGI-Claude` (SSH), poussé.
- **CUDA GPU fonctionnel** sur torch (build maison, 5 jours de mise au point) : voir section « Environnement GPU / CUDA » dans `CLAUDE.md` pour la séquence d'activation. Validé : `torch.cuda.is_available()` OK, matmul 2048×2048 4.3 ms GPU vs 10.5 s CPU.
- Conséquence pour les points 1-4 ci-dessous (« reste à faire ») : torch GPU est maintenant disponible, donc les tests/POC pourront tourner accélérés — plus seulement en CPU-only comme prévu au 2026-07-03.
- Reprise du fond du projet en cours (prochaine étape à détailler ici après la session).
- **Baseline validée end-to-end sous CUDA** : `pytest tests/` → 181 passed. `run_poc.py` tourne, checkpoint `.pkl` reprend correctement un run interrompu (vérifié sur 2 rounds consécutifs), `viewer.py` sert le dashboard (HTTP 200). Le workflow "2 commandes" (`run_poc.py --checkpoint ... --log ...` puis `viewer.py --log ...`) existe déjà et fonctionne — pas à reconstruire, juste à documenter/stabiliser.
- **Audit théorie↔code fait** (lecture complète de `SCL_fondements_mathematiques.md`, `SCL - Vision et Strategie.md`, `README v2.md`, tout `scl/*.py`) : socle (mémoires, module, module visuel, discriminateur, attention, création jumelée) fidèle à la théorie. Pas de résidu v1 (nettoyage déjà fait). Écart principal : couche cognitive avancée écrite/testée unitairement mais **non branchée** dans `boucle.py` (`recherche.py` A* ancrée, `allocation_attention.py` WFQ, `memoire_travail.py` multi-échelle, `disponibilite.py`, `statistiques.sprt_creation`, `simulateur.generer_contrefactuel/est_hors_distribution`), plus 4 divergences réelles (pas juste des trous) :
  1. `boucle.py` définit sa propre heuristique ad hoc de sélection d'action (`_scores_actions`) au lieu d'utiliser `decision_action.generer_actions_candidates` — contredit "l'orchestrateur compose, ne calcule jamais".
  2. ~~Création de module déclenchée par seuil fixe, pas par le SPRT prévu.~~ **CORRIGÉ** (commit `829e1a5`) : `boucle_temps_reel` accumule les échecs par point (`RegistreRupture.enregistrer_echec`), les évalue via `sprt_creation` avant toute création, teste la plausibilité (`discriminateur.evaluer_plausibilite`) puis tente une composition (`recherche.a_etoile_ancree`, nouvelle fonction `_tenter_composition`) avant de créer. Vérifié : 181 tests verts + POC longue durée (aucune création sur incident isolé, création confirmée quand H1 atteint après accumulation).
  3. Composition (recherche A* + plausibilité `discriminateur`) sautée : la boucle va direct de "rupture" à "création". — **traité par le point 2 ci-dessus** (même commit, la recherche A* ancrée est maintenant tentée avant la création quand le simulacre est plausible).
  4. Allocation d'attention fixe, pas le WFQ multi-fils prévu (`allocation_attention.allouer_capacite`). — reste à faire.
  1. `boucle.py` définit sa propre heuristique ad hoc de sélection d'action (`_scores_actions`) au lieu d'utiliser `decision_action.generer_actions_candidates` — contredit "l'orchestrateur compose, ne calcule jamais". — reste à faire.
  5. Fonction manquante (pas juste débranchée) : `consolidation_n_vers_un` (§9). — reste à faire.
- **Prochaine étape** : divergence #1 (heuristique ad hoc de sélection d'action) ou #4 (allocation d'attention WFQ).

## Fait

- **25 fonctions de la spec implémentées** (voir table de correspondance dans README.md) :
  - `scl/config.py` — hyperparamètres centralisés
  - `scl/logger.py` — audit JSONL : chaque action de chaque module journalisée
  - `scl/utils.py` — ajustement de dimensions, projections déterministes, 2-means
  - `scl/memoires.py` — TableBesoins, TableContexte, MémoireTampon, MémoireExceptions, RegistreCablage, RegistreRupture
  - `scl/module.py` — F2, F4, F5, F8, F9, F10, F11, F12 + croissance + copie détachée
  - `scl/orchestrateur.py` — F1, F3, F6, F14, F15, F16, F23 + gate contrastive (F22) + priorités apprises
  - `scl/graphe.py` — F7 (perception/imagination/fusion), F13, F17, F18, F19, F20 + garde-fous création
  - `scl/monde.py` — grille infinie procédurale, accel ±1 X/Y, v_max 2, perception 10×10×3, sucres/bâtons
  - `scl/inne.py` — graphe de naissance (vision, proprio, intégration, action, réflexe frein verrouillé)
  - `scl/boucle.py` — F21, F22, F24, F25
  - `run_poc.py` — CLI
- **24 tests de robustesse écrits** (`tests/test_robustesse.py`), un test par TEST de la spec.
- **Vérifié dans la sandbox Linux** (sans torch) :
  - compilation de tous les fichiers (`py_compile`) : OK
  - présence des 24 tests et de toutes les fonctions attendues (analyse AST) : OK
  - monde simulé exécuté 200 steps réels (perception 3×10×10, collisions, besoins, 603 actions d'audit) : OK

## Reste à faire (chez toi, WSL Ubuntu)

1. `pip install torch numpy pytest` (torch CPU suffit) — non installable dans ma sandbox (limite de temps réseau).
2. `python3 -m pytest tests/ -v` — les 24 tests n'ont **pas encore été exécutés** ; la partie torch est vérifiée statiquement seulement. Attends-toi à d'éventuels ajustements de seuils (tests 5, 16, 20 dépendent de valeurs d'initialisation aléatoire).
3. `python3 run_poc.py --jours 3 --steps 500` — premier run d'observation ; auditer `scl_audit.jsonl`.
4. Calibration empirique probable : `seuil_regime_2`, `seuil_rupture`, `seuil_succes` (les erreurs MSE réelles dépendent de l'échelle des latents).

## Écarts documentés (choix, pas de refonte — cf. README §Choix d'implémentation)

- Cibles F9/F10 : prédiction temporelle locale (latent t−1 → latent t / input t), auto-supervisée, strictement locale.
- `meilleure_projection` : échafaudage inné fournissant la cible d'aligner_action (rollout corporel 3 pas). La mécanique F8 est respectée à la lettre.
- Incertitude en imagination (F16) : proxy = norme moyenne des sorties imaginées.
- `W_sens` biaisé bottom_up à l'initialisation (sinon routage aléatoire à la naissance).
- Découpe (F18) : noyau = copie exacte + amovible additif porté par `graphe.compositions` (pas d'arête parasite), garantit TEST 18.

## Incident mineur

La synchro du dossier vers la sandbox s'est figée sur `scl/graphe.py` (copie tronquée côté Linux). Le fichier sur ton disque (`D:\IA\AGI-Claude\scl\graphe.py`) est complet et correct — vérifié en le relisant et en compilant une copie reconstruite. Rien à faire de ton côté.
