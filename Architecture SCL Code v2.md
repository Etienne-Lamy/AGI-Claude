# Architecture SCL — Code v2 (document cadre)

Document de référence, indépendant de tout état de code antérieur. Décrit l'architecture complète requise pour réaliser `SCL_fondements_mathematiques.tex` (v6) : chaque module, chaque fonction, son rôle, sa référence théorique, ses entrées/sorties, ses dépendances. Utilisable pour recréer le code intégralement ou pour l'auditer/le modifier — dans les deux cas, ce document est la référence, pas le code. Aucun code ici.

**Format par fonction** : `nom` — rôle · §SCL · entrées → sorties · appelle · appelé par.

---

## 0. Principes non négociables

Contraintes qui s'appliquent à toute fonction de ce document, rappelées ici une fois pour ne pas être répétées partout :

- Aucun gradient ne traverse une frontière de module (§2.1). Toute entrée inter-modules est détachée avant usage.
- Verrouillage = plancher, jamais plafond (§1.4). Une mise à jour ultérieure d'un module verrouillé est acceptée si elle passe le test de non-infériorité, jamais interdite par principe.
- Aucune indexation par horaire absolu (§11.1, §12). Toute mémoire temporelle est indexée par offset relatif à l'instant courant.
- Un seul discriminateur $D_\phi$, partagé par tous les modules et par le rejeu nocturne — jamais un discriminateur par module (§1.3, §5).
- Toute composition choisie par l'orchestrateur doit être ancrée à un point de comparaison vérifiable — réel brut ou module intermédiaire déjà certifié (§7.4). Sans cela, aucune composition n'est évaluable.
- Toute décision structurelle (croissance, branchement d'opérateur, composition) est discrète ; seuls les coefficients scalaires sont optimisés par gradient continu (§14).
- Le vecteur de besoins $b_t$ gouverne l'action par un seul besoin dominant à la fois (argmax + hystérésis), jamais par mélange pondéré continu (§15.3).
- Chaque mécanisme porte une étiquette de statut épistémique : pilier démontré, hypothèse à tester, ou décision de conception (grille de lecture de la v6, à répercuter dans les commentaires de code).

---

## 1. Vue d'ensemble des modules

| Module (fichier) | Rôle | §SCL principal |
|---|---|---|
| `config.py` | constantes et hyperparamètres | Annexe B |
| `utils.py` | utilitaires génériques, SPRT séquentiel générique | transverse |
| `logger.py` | audit exhaustif, une ligne par événement | transverse |
| `checkpoint.py` | persistance de l'état complet | transverse |
| `monde.py` | environnement simulé (2D, capteurs, actions) | cadre expérimental |
| `memoires.py` | besoins, contexte, tampon jour, exceptions, registres | §1.1, §1.4, §8, §9 |
| `module.py` | unité d'apprentissage locale $(E,G)$, condensateur, $\pi_i(x)$, accumulateur $\bar g_i$ | §1.3, §1.4 |
| `module_visuel.py` | module sensoriel par défaut, bootstrap auto-supervisé | §1.3 |
| `discriminateur.py` | $D_\phi$ partagé, classification generative-contrastive | §5 |
| `disponibilite.py` | disponibilité anticipée, logique d'acceptation des mises à jour | §1.4 |
| `graphe.py` | structure, croissance gouvernée, rejet gouverné, fragmentation, découpe, localisation, création jumelée, non-infériorité, consolidation, drift, atrophie | §1.2, §2, §4.6, §9 |
| `simulateur.py` | $S_{\text{new}}$, mémoire épisodique générative | §4.5, §10.2 |
| `attention.py` | Set Transformer, $T_t$, Pointer Network, triplet, exécution parallèle, types, trace auto-référentielle, activation creuse, sommeil, accumulateur orchestrateur, apprentissage REINFORCE | §10 (en entier) |
| `recherche.py` | A\*, heuristique apprise $V_\psi$, ancrage, A\* ancrée | §7 |
| `credit.py` | regret de composition, décomposition du crédit | §10.8 |
| `allocation_attention.py` | allocation dynamique de la capacité (WFQ) | §13 |
| `memoire_travail.py` | tampon à décalage relatif, familles d'horizon, fenêtre glissante, palier de sommeil | §11, §12 |
| `operateurs_natifs.py` | primitives de calcul niveau ISA | §14 |
| `decision_action.py` | fusion pondérée, récompense intrinsèque, besoin dominant | §15 |
| `statistiques.py` | SPRT (surprise, création, drift), contrôle FDR | §4, M1, M10 |
| `inne.py` | construction du graphe de naissance | bootstrap |
| `boucle.py` | boucle temps réel, cycle nocturne, boucle principale | orchestration temporelle globale |

---

## 2. Structures de données fondamentales

- **`Module`** : paire $(E,G)$ + condensateur global $c$ (reco/gen) + fiabilité contextuelle $\pi(x)$ + accumulateur de gradient $\bar g$ + canal réinjecté $d^{\text{réinj}}$ + embedding $e\in\mathbb R^{\dim_{\text{emb}}}$ + statut (actif/en_test/verrouillé/abandonné/provisoire).
- **`Simulateur`** ($S_{\text{new}}$) : associé 1:1 à un module créé par le pipeline de création ; tête hétéroscédastique obligatoire ; refabrique un contexte d'échec (brut ou latent).
- **`Triplet`** : $(\text{src}, \text{op}, \text{cib})$, trois pointeurs sur les indices courants de $T_t$.
- **$T_t$** : ensemble $\mathcal F_t^{\text{ctx}} \cup \mathcal F_t^{\text{ptr}}$ — contexte non pointable (condensateurs, fiabilités, besoins, verrous) et éléments pointables (capteurs, latents de modules, trace).
- **`TableBesoins`** : vecteur $b_t$, besoin dominant courant $k_t$, marge d'hystérésis.
- **`RegistreCablage`, `RegistreRupture`, `RegistreDisponibilite`, `RegistreProvenance`** : historiques structurels (câblage, cooldown, échantillon varié $\mathcal W_i(t)$, bit réel/imaginé).
- **`Config`** : dictionnaire de constantes scalaires, aucune catégorie sémantique.

---

## 3. `config.py`

Aucune fonction ; dictionnaire de constantes couvrant l'intégralité de l'Annexe B de la v6 : dimensions ($\dim_{\text{emb}}$, $\dim_{\text{op}}$, $d_{\text{model}}$, nombre de têtes, $W$, $K$, familles d'horizon, nombre de types $\tau$, $d^{\text{réinj}}$), seuils ($\epsilon_s$, $\epsilon_\sigma$, $\Delta$ non-infériorité, quantile $\chi^2$, $(\alpha,\beta)$ SPRT surprise et création, $\alpha$ FDR), moments ($\beta$ jour/nuit des accumulateurs $\bar g_i$ et $\bar g_{\text{orch}}$), plafond de ratio imaginé/réel, marge d'hystérésis $\delta$ des besoins, catalogue d'opérateurs natifs.

---

## 4. `utils.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `ajuster_dim` | pad/tronque un vecteur à une dimension cible | transverse | vecteur, n → vecteur | — | quasi tout le code |
| `projeter` | projection aléatoire déterministe (comparaison de flux hétérogènes sans apprentissage) | transverse | vecteur, n → vecteur | — | `attention.py`, `discriminateur.py` |
| `kmeans2` | 2-means minimal | §2.2 (découpe) | tenseur → labels, c0, c1 | — | `graphe.decouper_module` |
| `separation_claire` | test de séparation inter/intra-cluster | §2.2 | tenseur, labels, centres → bool | — | `graphe.decouper_module` |
| `pente` | régression linéaire simple sur une série | §1.4 (plateau de progrès) | liste → pente | — | `disponibilite.py`, `module.py` |
| `sprt_sequentiel` | test séquentiel générique $\Lambda_n$ sur un flux de rapports de vraisemblance | §4.3 | flux, $(\alpha,\beta)$ → décision {continuer, $H_0$, $H_1$} | — | `statistiques.py` (3 usages : surprise, création, drift) |

---

## 5. `logger.py`

| Fonction | Rôle | Entrées → Sorties | Appelé par |
|---|---|---|---|
| `AuditLogger.__init__` | instancie le journal (fichier + tampon mémoire) | chemin, options → instance | `configurer` |
| `log` | journalise un événement structuré (acteur, action, détails) | acteur, action, détails → (écriture) | toutes les fonctions du système, sans exception |
| `log_verbeux` | journalisation fine, activée seulement en mode verbeux | idem | fonctions à haute fréquence (forwards, gates) |
| `set_temps` | fixe le contexte temporel courant (jour, step) | jour, step → (mutation) | `boucle.py` |
| `filtrer` | interroge le tampon mémoire par acteur/action | acteur, action → liste d'enregistrements | tests, dashboards |
| `configurer` / `obtenir` / `est_verbeux` | gestion du logger global | — | point d'entrée du programme |

Exigence : chaque fonction de ce document journalise son résultat significatif via `log` ou `log_verbeux` — pas d'action silencieuse sur l'état structurel.

---

## 6. `checkpoint.py`

| Fonction | Rôle | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|
| `sauvegarder` | sérialise l'état complet (graphe, monde, besoins, registres, accumulateurs, mémoire de travail, simulateurs) | chemin, état → (écriture atomique) | — | `boucle.main_loop` |
| `charger` | désérialise et reconstruit l'état | chemin → état complet | — | `boucle.main_loop` |
| `existe` | teste la présence d'un checkpoint | chemin → bool | — | `boucle.main_loop` |

---

## 7. `monde.py`

Environnement de test — hors du corps théorique SCL, mais l'interface que tout le reste consomme.

| Fonction | Rôle | Entrées → Sorties | Appelé par |
|---|---|---|---|
| `Monde.__init__` | construit le monde procédural (grille infinie, génération par chunks) | graine → instance | `inne.construire_graphe_inne`, `boucle.main_loop` |
| `_objets_chunk` | génère un chunk déterministe (sucres, bâtons) | (cx,cy) → dict objets | `objet_en` |
| `objet_en` | objet présent à une position | (x,y) → type d'objet ou rien | `_frame`, `appliquer_action` |
| `appliquer_action` | applique une accélération, détecte les collisions | accélération → événements | `boucle.boucle_temps_reel` |
| `_frame` | rend le champ visuel courant (niveaux de gris, corps visible) | — → tenseur | `percevoir` |
| `percevoir` | construit le contexte perceptif (vision, proprioception, position) | — → dict | `boucle.boucle_temps_reel` |
| `objets_visibles` | positions relatives des objets dans le champ courant | — → (sucres, bâtons) | `decision_action.py`, `attention.py` (génération d'actions, §15.3) |

---

## 8. `memoires.py`

### 8.1 Besoins et contexte (§1.1)

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `TableBesoins.__init__` | vecteur de besoins $b_t$, extensible au-delà du minimum (faim, ennui) | §1.1 | — → instance | — | `boucle.main_loop` |
| `TableBesoins.mettre_a_jour` | applique décroissance naturelle + événements du monde | §1.1 | événements, contexte moteur → (mutation) | — | `boucle.boucle_temps_reel` |
| `TableBesoins.besoin_dominant` | $k_t = \arg\max_k b_t[k]$ avec hystérésis de marge $\delta$ (Schmitt-trigger) | §15.3 | $b_t$, $k_{t-1}$ → $k_t$ | — | `decision_action.priorisation_besoin_dominant` |
| `TableContexte.__init__` / `mettre_a_jour` | état global "normal"/"choc" (inhibe la création structurelle hors contexte normal) | §4 (garde-fou de portée) | besoins → (mutation) | — | `boucle.py` |

### 8.2 Mémoire tampon et exceptions (§8)

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelé par |
|---|---|---|---|---|
| `MemoireTampon.ajouter_reco` / `ajouter_gen` | enregistre une tentative de la journée (rejouée la nuit) | §8.1 | module_id, input, cible, erreur, t, contexte → (mutation) | `boucle.boucle_temps_reel` |
| `MemoireTampon.pour_point` | tentatives liées à un module ou point de rupture | §8.1 | id → tentatives reco, gen | `graphe.py`, `boucle.cycle_nocturne` |
| `MemoireTampon.clear` | purge en fin de nuit | §8.1 | — → (mutation) | `boucle.cycle_nocturne` |
| `MemoireExceptions.ajouter` / `non_resolues` | situations non résolues, revisitées la nuit | §4.5 (esprit proche) | contexte, erreur, t → (mutation) / — → liste | `boucle.py` |

### 8.3 Registres structurels (§8, §9, §1.4)

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelé par |
|---|---|---|---|---|
| `RegistreCablage.append` | historise une insertion structurelle (rupture, découpe, exploratoire, création) | §8, §9 | module_id, point, contexte, t, type → entrée | `graphe.py`, `attention.py` |
| `RegistreRupture.peut_creer` / `marquer_abandon` | cooldown de création par point de rupture | §1.4 (esprit) | point, t → bool / (mutation) | `graphe.creer_module_candidat` |
| `RegistreDisponibilite.ajouter` | ajoute un contexte à l'échantillon varié $\mathcal W_i(t)$ (dédoublonné par diversité, pas par consécutivité) | §1.4 | module_id, contexte → (mutation) | `boucle.boucle_temps_reel` |
| `RegistreDisponibilite.echantillon` | renvoie $\mathcal W_i(t)$ courant | §1.4 | module_id → liste de contextes | `disponibilite.disponibilite_anticipee` |
| `RegistreProvenance.marquer` | attache le bit {réel, imaginé} à un exemple stocké | §8.3, M6 | exemple, provenance → (mutation) | `simulateur.generer_contrefactuel` |
| `RegistreProvenance.purger` | supprime en cascade tous les exemples liés à un module purgé | §8.3, M6 | module_id → n supprimés | `graphe.py` (désuétude, §2.3) |
| `RegistreProvenance.ratio_lot` | plafonne le ratio imaginé/réel d'un lot d'entraînement | §8.3, M6 | lot → lot borné | `attention.entrainer_pointeurs` |

---

## 9. `module.py`

Unité d'apprentissage strictement locale. Aucun gradient n'en sort.

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `Module.__init__` | instancie $(E,G)$, condensateur, embedding, canal réinjecté $d^{\text{réinj}}$ | §1.2, §1.3 | dimensions → instance | `_init_poids` | `graphe.py`, `inne.py` |
| `_init_poids` / `_rebuild_optimizers` | construction des poids et optimiseurs internes | §1.3 | — | — | `__init__`, `grandir` |
| `parametres_reco` / `parametres_gen` / `parametres` | accès aux paramètres | — | — → liste | — | entraînement, snapshots |
| `etat_dict` / `charger_etat` | snapshot / restauration des poids (non-dégradation, copie) | §9 | — | — | `graphe.py`, `copier_module` |
| `forward_reconnaissance` | encodeur $E$ | §1.3 | input → latent | — | ~tout |
| `forward_generation` | décodeur $G$ | §1.3 | latent → output | — | ~tout |
| `aligner_action` | recherche de commande motrice sur le latent d'entrée, poids figés | §1.3 | projection souhaitée → commande | `forward_generation` | `decision_action.py` |
| `chercher_latent_predictif` | cible prédictive : latent qui aurait le mieux prédit l'observation via le décodeur figé | §1.3 | input précédent, cible → latent | `forward_reconnaissance/generation` | `entrainer_module_reco`, `simulateur.py` |
| `entrainer_module_reco` / `entrainer_module_gen` | mise à jour locale (BCD), incorporation à l'accumulateur $\bar g$ avant application | §1.3, §2.1 | input/latent, cible → erreur | `incorporer_gradient`, `forward_*` | `boucle.py`, `graphe.py` |
| `fiabilite_contextuelle` | $\pi(x) = 1-\hat{\mathcal L}^{\text{relative}}(x)$, fiabilité indexée par contexte (pas par instant) | §1.4 | contexte x → $\pi\in[0,1]$ | — | `attention.construire_T_t`, `graphe.localiser_point_branchement`, `disponibilite.py` |
| `incorporer_gradient` | met à jour $\bar g$ (moment, $\beta$ différent jour/nuit) | §1.3 | $\nabla$, phase → (mutation) | — | `entrainer_module_reco/gen`, `boucle.cycle_nocturne` |
| `mettre_a_jour_condensateurs` | condensateur global + verrouillage asymétrique (plancher, jamais plafond) | §1.4 | erreurs → (mutation) | `disponibilite.logique_acceptation` | `boucle.boucle_temps_reel` |
| `detecter_saturation` | double signal gradient faible + erreur mauvaise | §2.3 (rejet gouverné, esprit) | — → (bool, bool) | — | `boucle.boucle_temps_reel` |
| `grandir` | croissance dimensionnelle conditionnée au gain (MDL, §3.3) | §3.3 | voie, pas → bool | `_rebuild_optimizers` | `boucle.faire_croitre_si_gain` |
| `evaluer_reco` / `evaluer_gen` | évaluation sans apprentissage (support du test de non-infériorité) | §9 | tentatives → erreur moyenne | `forward_*` | `graphe.test_non_inferiorite` |
| `copier_module` | copie structurelle exacte, poids détachés | §9 (fragmentation) | module, id → Module | `charger_etat` | `graphe.fragmenter_module`, `decouper_module` |

---

## 10. `module_visuel.py`

Module sensoriel par défaut : donné a priori, pas découvert par l'orchestrateur. Interface identique à `Module` (mêmes entraînements, condensateurs, verrous, recherche de latent prédictif) ; seule la paramétrisation change (biais inductif adapté à la structure spatiale de l'entrée).

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `ModuleVisuel.__init__` | encodeur/décodeur dimensionné pour la résolution du champ visuel | §1.3 | résolution → instance | `_init_poids` | `inne.construire_graphe_inne` |
| `entrainer_masque` | auto-supervision par reconstruction masquée (JEPA) : masque aléatoire appliqué à l'entrée, cible = reconstruction complète | §1.3 | champ visuel → erreur | `forward_reconnaissance/generation` | `boucle.boucle_temps_reel` |
| `chercher_latent_predictif` | comme `module.py`, spécialisé pour l'espace latent visuel (initialisations multiples si non convexe) | §1.3 | input, cible → latent | `forward_*` | `boucle.py`, `pilote/attention.py` |

---

## 11. `discriminateur.py`

$D_\phi$ partagé — jamais un par module.

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `Discriminateur.__init__` | réseau unique, partagé par tout le système | §5, §1.3 | dimension → instance | — | `inne.py` (instancié une fois) |
| `evaluer_plausibilite` | $D_\phi(x)$, probabilité que $x$ appartienne à la réalité | §5, §4.5 | x → probabilité | — | `simulateur.py`, `memoire_travail.py`, `graphe.py` (pipeline de création) |
| `entrainer_contrastif` | NCE : un positif réel contre $N$ négatifs générés | §5 | $x^+$, $\{x_j^-\}$ → (mutation) | — | `boucle.cycle_nocturne` |
| `attenuer_soft` | atténuation douce (shrinkage), jamais un masque à zéro | §5 | poids, rang → poids atténués | — | `entrainer_contrastif` |

---

## 12. `disponibilite.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `disponibilite_anticipee` | teste $|\rho_i(t)|<\epsilon_s$ (plateau de progrès) et $\text{Var}(\hat\sigma_i)<\epsilon_\sigma$ sur l'échantillon varié $\mathcal W_i(t)$ | §1.4 | module → bool | `utils.pente`, `memoires.RegistreDisponibilite.echantillon` | `graphe.ajouter_module` (entrée dans $\mathcal F_t^{\text{ptr}}$) |
| `logique_acceptation` | accepte/rejette une mise à jour selon la variation de $\pi_i(x)$ ; si dégradation, entraîne le prédicteur de $\pi$ à signaler le contexte plutôt que de mettre à jour $\theta_i$ | §1.4 | module, (x,y) → décision | `module.fiabilite_contextuelle`, `graphe.test_non_inferiorite` | `module.entrainer_module_reco/gen` |

---

## 13. `graphe.py`

Structure pérenne de modules et opérations structurelles.

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `Graphe.__init__` / `ajouter_module` / `retirer` | structure du graphe | §1.2 | — | — | `inne.py`, ~tout |
| `parents` / `enfants` / `ordre_topologique` | navigation structurelle | — | — | — | `forward_graphe`, `localiser_point_branchement` |
| `erreur_globale` / `noter_erreur` | erreur agrégée | §4 (signal global) | — | — | `boucle.py` |
| `forward_graphe` | passe avant : perception / imagination / fusion, exécutée par macro-pas de largeur $W$ | §1.2, §10.3, §15.1 | contexte, mode → outputs | `attention.macro_pas` | `boucle.boucle_temps_reel` |
| `croissance_gouvernee` | accepte une croissance ssi $\Phi_t$ décroît strictement ($\varepsilon$) à catalogue $A_t$ fixé (monotonie par morceaux, pas de garantie globale au changement de support) | §2.2 | tentative de croissance → accept/reject | — | `module.grandir`, `graphe.decouper_module` |
| `rejet_gouverne` | test de rapport de vraisemblance (Wilks, $\chi^2$ asymptotique) pour conserver/rejeter un module à gating conditionnel | §2.3 | module, contexte → conserver/rejeter | — | `boucle.cycle_nocturne` |
| `localiser_point_branchement` | premier module au sens du flux dont $\pi_i(x)$ s'effondre alors que tous ses antécédents directs restent hauts ; si c'est le capteur lui-même, branchement en tête | §4.6 | contexte x → point (ou "capteur") | `module.fiabilite_contextuelle` | `boucle.boucle_temps_reel` |
| `creer_module_candidat` | point d'entrée de la création jumelée : instancie le module et son simulateur associé | §4.5(5), §10.2 | point, dimensions, contexte d'échec → (Module, Simulateur) | `simulateur.Simulateur.__init__` | `boucle.py` (pipeline §4.5, étape 5) |
| `fragmenter_module` | module effondré → règle générale (dégelée) + exception (en test), competing_ids croisés | §9 | module, registre, contexte, t → (règle, exception) | `copier_module` | `boucle.py` |
| `decouper_module` | sépare noyau (copie exacte) + amovible additif sur la variable discriminante | §9 | module, tampon, t → (noyau, amovible) | `kmeans2`, `separation_claire` | `boucle.py` |
| `sortie_composee` | combinaison additive noyau/amovible, gate=0 ⇒ module d'origine inchangé | §9 | noyau, amovible, x, gate → tenseur | `forward_reconnaissance` | `valider_decoupe` |
| `test_non_inferiorite` | test formel réutilisé partout où une comparaison A/B doit être tranchée sans dégrader : verrouillage, découpe, consolidation, certification d'un module créé | §9 | échantillons appariés, $\Delta$, $\alpha$ → accepter/rejeter | — | `valider_decoupe`, `module.mettre_a_jour_condensateurs`, `simulateur.py`, `disponibilite.logique_acceptation` |
| `valider_decoupe` | décision intégrer / abandonner / fusionner-retour, via `test_non_inferiorite` | §9 | noyau, amovible, tampon → décision | `test_non_inferiorite` | `boucle.cycle_nocturne` |
| `consolidation_n_vers_un` | remplace $n$ modules satellites par un module partagé si le critère MDL le justifie, même test que la découpe | §9 | ensemble de modules → module unique ou statu quo | `test_non_inferiorite` | `boucle.cycle_nocturne` |
| `controle_multiplicite` | applique FDR (Benjamini-Hochberg) ou budget $\alpha$ journalier à l'ensemble des tests de non-infériorité exécutés dans la fenêtre | M10 | liste de tests du jour → seuil ajusté | `statistiques.controle_fdr` | `boucle.cycle_nocturne` |
| `recalage_plancher_drift` | re-mesure le plancher $c_i^{\min}$ d'un module certifié si le SPRT de nouveauté conclut à un drift durable sur son domaine | M10 | module, résultat SPRT → (mutation) | `statistiques.sprt_drift` | `boucle.cycle_nocturne` |
| `atrophier` | retire un module mûr n'ayant jamais acquis de certitude ; si le module est provisoire et non confirmé, purge en cascade ses exemples de provenance | §2.3, §8.3(M6) | module → (mutation) | `memoires.RegistreProvenance.purger` | `boucle.cycle_nocturne` |
| `committer_chemin` | ajoute les arêtes d'un chemin exploratoire validé en imagination | §7, §8.1 | chemin → nouvelles arêtes | — | `boucle.py` |

---

## 14. `simulateur.py`

$S_{\text{new}}$ — mémoire épisodique générative, pas un stockage brut de l'épisode.

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `Simulateur.__init__` | instancié au point de branchement localisé, tête hétéroscédastique obligatoire | §10.2 | dimensions, contexte d'échec → instance | — | `graphe.creer_module_candidat` |
| `refabriquer` | régénère un contexte stocké (brut de capteurs ou latent d'un module amont certifié), potentiellement purement latent | §10.2, §10.7 | $z_{\text{stocké}}$ → $(\mu,\Sigma)$ | — | `memoire_travail.PalierSommeil.recuperer` |
| `generer_contrefactuel` | rollout de chemins non empruntés à partir d'un succès réel — dream/nightmare augmentation | §8.3 | chemin réel $\gamma^+$ → $\gamma^-$ | `discriminateur.evaluer_plausibilite` | `boucle.cycle_nocturne` |
| `est_hors_distribution` | signale que le verdict de $D_\phi$ doit être étiqueté hypothèse, pas pilier, dans ce cas d'usage (radicalement neuf) | §10.2 | contexte → bool | `discriminateur.evaluer_plausibilite` | `graphe.py` (pipeline §4.5) |

---

## 15. `attention.py`

Le cœur de l'orchestrateur : Set Transformer en entrée, Pointer Network en sortie. Chantier le plus dense du système — regrouper ici tout §10.

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `construire_T_t` | assemble $T_t = \mathcal F_t^{\text{ctx}} \cup \mathcal F_t^{\text{ptr}}$ à partir du graphe et du contexte courant | §10.1, §1.2 | graphe, contexte → $T_t$ | `module.fiabilite_contextuelle` | `boucle.boucle_temps_reel` |
| `SetTransformer.encoder` | encode $T_t$ en une représentation invariante par permutation | §10.1 | $T_t$ → représentation | — | `PointerNetwork.decoder` |
| `PointerNetwork.decoder` | émet un triplet (src, op, cib) par pointeurs softmax sur les indices courants, jeton `[NEW]` et opérateur `id` inclus | §10.2 | représentation → triplet | `masque_compatibilite_type`, `critere_arret_fil` | `boucle.boucle_temps_reel` |
| `masque_compatibilite_type` | applique $-\infty$ aux indices dont le type $\tau$ est incompatible avec l'opérateur pointé, avant softmax | §10.2, §10.5 | logits, types → logits masqués | — | `PointerNetwork.decoder` |
| `transfert_inter_dimensionnel` | autorise le transfert d'un opérateur appris sur un type vers un autre type partageant une propriété structurelle déclarée | §10.5 | opérateur, type source, type cible → autorisation | — | `PointerNetwork.decoder` |
| `executer_triplet` | applique l'opérateur pointé, écrit le résultat à l'emplacement pointé (module existant, $T_{t+1}$, ou port moteur) | §10.2, §10.3 | triplet → résultat | module ciblé | `macro_pas` |
| `macro_pas` | lot de $w\le W$ triplets simultanés ; `id` pour toute dépendance non prête (bulle de pipeline) | §10.3 | $T_t$ → lot de résultats | `executer_triplet` | `boucle.boucle_temps_reel` |
| `critere_arret_fil` | arrête un fil de décodage : incertitude propagée, aucun candidat valable, port terminal atteint, ou profondeur maximale en dernier recours | §10.4 | fil → bool | — | `PointerNetwork.decoder` |
| `trace_autoreferentielle` | réinjecte $\text{trace}_{t-1}$ comme élément de $\mathcal F_t^{\text{ptr}}$ | §10.6 | trace précédente → élément de $T_t$ | — | `construire_T_t` |
| `activation_creuse` | sélectionne les $w\le W$ modules actifs parmi $|A_t|\gg W$ disponibles | §10.7 | $A_t$ → sous-ensemble actif | — | `macro_pas` |
| `entrainer_pointeurs` | REINFORCE sur la trajectoire de triplets, baseline = regret de composition | §10.2(M8) | trajectoire, regret → (mutation poids du pointeur) | `credit.regret_composition` | `boucle.cycle_nocturne` |
| `accumulateur_orchestrateur` | met à jour $\bar g_{\text{orch}}$, distinct de $\bar g_i$, alimenté par le résidu de pertinence de la composition | §10.8 | résidu, phase → (mutation) | `credit.regret_composition` | `boucle.py` |

---

## 16. `recherche.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `a_etoile` | recherche $f(n)=g(n)+h(n)$, dégénère en recherche exhaustive si $V_\psi\equiv 0$ | §7.1, §7.3 | nœud départ, objectif → chemin | `v_psi` | `graphe.py` (pipeline §4.5) |
| `v_psi` | heuristique apprise, non garantie admissible | §7.2 | nœud → valeur | — | `a_etoile` |
| `entrainer_v_psi` | mise à jour TD (Robbins-Monro) | §7.2 | transition (n, r, n') → (mutation) | `credit.regret_composition` (référence commune) | `boucle.cycle_nocturne` |
| `ancrer_composition` | pousse la cascade de générateurs jusqu'à un point de comparaison vérifiable (réel ou module certifié) ; arrêt anticipé via `test_non_inferiorite` | §7.4 | composition candidate → point de comparaison | `graphe.test_non_inferiorite` | `a_etoile_ancree` |
| `a_etoile_ancree` | A\* dont $g,h$ sont évalués sur le point d'ancrage à chaque nœud ; fusion réel/imaginé pondérée par la profondeur | §7.5 | nœud → $(g,h)$ | `ancrer_composition`, `v_psi`, `decision_action.fusion_ponderee` | `graphe.py` (pipeline §4.5) |

---

## 17. `credit.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `regret_composition` | $\hat{\mathcal L}_{\text{choisi}}(x,y) - \min_{j\in\text{candidats}(x)}\hat{\mathcal L}_j(x,y)$, rejeu contrefactuel comme baseline | §10.8(M7) | x, y, choisi, candidats → regret | `module.evaluer_reco/gen` | `attention.entrainer_pointeurs`, `recherche.entrainer_v_psi` |
| `approx_regret_jour` | approxime le regret en journée via $V_\psi$ des alternatives non prises, sans rejeu complet | §10.8(M7) | contexte → regret approché | `recherche.v_psi` | `boucle.boucle_temps_reel` |
| `rejeu_contrefactuel_nocturne` | rejoue tous les candidats disponibles (poids figés) contre la cible réelle pour un échantillon de contextes | §8.1 | échantillon de contextes → jeu de résidus par candidat | `module.evaluer_reco/gen` | `boucle.cycle_nocturne`, `regret_composition` |
| `amorcage_creation` | injecte immédiatement le premier exemple positif dans le jeu d'apprentissage du gating à la création d'un module | §8.2 | module créé, contexte de création → (mutation jeu d'apprentissage) | — | `graphe.creer_module_candidat` |

---

## 18. `allocation_attention.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `allouer_capacite` | répartit le budget $W$ entre fils/rôles concurrents proportionnellement à leur urgence $u_k(t)$ (Weighted Fair Queueing) | §13 | urgences des fils → allocation $w_k(t)$ | — | `attention.macro_pas` |
| `urgence_fil` | dérive l'urgence d'un fil à partir des signaux de besoin et de surprise déjà définis | §13 | fil, $b_t$, résidu de surprise → urgence | `memoires.TableBesoins`, `statistiques.sprt_surprise` | `allouer_capacite` |
| `role_creation` | déclare la création de module comme fil concurrent explicite, sous le garde-fou câblé, avec partage jour (minimum viable) / nuit (entraînement complet via $S_{\text{new}}$) | §13(M5) | contexte de création → part du budget $W$ | `urgence_fil` | `boucle.boucle_temps_reel`, `cycle_nocturne` |

---

## 19. `memoire_travail.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `TamponRelatif.__init__` | tampon circulaire de taille $2K+1$, indexé par offset relatif $\delta\in\{-K,\dots,K\}$, jamais par horaire absolu | §11.1 | $K$ → instance | — | `boucle.py` (un par flux), `attention.py` |
| `TamponRelatif.decaler` | décale tous les offsets d'un tick à chaque pas ; le contenu sortant à $-K$ est archivé ou perdu | §11.1 | nouvelle observation → (mutation) | — | `boucle.boucle_temps_reel` |
| `hierarchie_deux_vitesses` | tête rapide (bornée par $W$) et palier lent plus grand, analogie registre/mémoire principale | §11.1 | — | — | `TamponRelatif.__init__` |
| `FamilleHorizon.emettre` | émet une prédiction à l'offset relatif $+h^{(k)}$, à cadence propre à la famille (court/moyen/long) | §12 | module, $h$ → (mutation tampon) | `module.forward_generation` | `boucle.boucle_temps_reel` |
| `FamilleHorizon.maturer` | quand l'offset atteint 0, compare au réel ; chaque prédiction sert deux fois (décision à l'émission, évaluation à la maturation) | §12 | tampon → résidu | `statistiques.sprt_surprise` | `boucle.boucle_temps_reel` |
| `fenetre_glissante_continue` | pas de replanification complète : décale la fenêtre, le masquage se lève tick par tick au rythme où le réel remplace le prévu | §12 | tampon, réel courant → (mutation) | — | `FamilleHorizon.maturer` |
| `PalierSommeil.stocker` | maintient un contexte problématique via l'opérateur `id` (idle/NOP), hors mémoire de calcul rapide | §10.7 | contexte non résolu → (mutation) | — | `graphe.py` (pipeline §4.5, étape 5) |
| `PalierSommeil.recuperer` | récupère un contexte stocké pour rejeu nocturne, sous précondition de reconstruction jugée suffisante par $D_\phi$ | §10.7 | — → contexte ou refus | `discriminateur.evaluer_plausibilite`, `simulateur.refabriquer` | `boucle.cycle_nocturne` |

---

## 20. `operateurs_natifs.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelé par |
|---|---|---|---|---|
| `primitives_isa` | catalogue figé et volontairement petit : addition, multiplication, comparaison, décalage de bits, lecture/écriture mémoire, constructions de boucle | §14 | — → liste d'opérateurs | `attention.construire_T_t` (enrichissement du vocabulaire $\Omega$) |
| `garde_fou_domaine` | protège les opérations à singularité (division, etc.) par clamp/epsilon | §14 | opération, opérandes → résultat protégé | exécution des primitives |
| `composition_sure` | distingue les compositions sûres (exposants entiers fixes, linéaires dans les paramètres) des compositions risquées (exposant appris, divisions en cascade) | §14 | composition candidate → catégorie | `attention.executer_triplet` |

---

## 21. `decision_action.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `fusion_ponderee` | combinaison continue, pondérée par confiance, de la perception réelle et de la génération prédite — jamais un remplacement binaire | §15.1 | perception, prédiction, confiance → flux fusionné | — | `graphe.forward_graphe` (mode fusion) |
| `recompense_intrinseque` | $r^{\text{intrinsèque}}_t = L_{\text{total},i}(t-1)-L_{\text{total},i}(t)$, unifie baisse d'erreur et baisse de complexité | §15.2 | historique de $L_{\text{total}}$ → récompense | — | `boucle.py` (signal d'apprentissage) |
| `priorisation_besoin_dominant` | sélectionne l'action selon le seul besoin actif $k_t$ ; garde-fou câblé (réflexe de douleur) prioritaire, évalué avant | §15.3 | $k_t$, actions candidates → action | `memoires.TableBesoins.besoin_dominant` | `boucle.boucle_temps_reel` |
| `generer_actions_candidates` | énumère les actions depuis $\mathcal A$ (discret, borné), les évalue par rollout dans les tampons multi-échelle via la fusion pondérée | §15.3(M12) | contexte, $\mathcal A$ → actions scorées | `memoire_travail.FamilleHorizon`, `fusion_ponderee` | `priorisation_besoin_dominant` |
| `reflexe_cable` | garde-fou non appris, court-circuite toute sélection par besoin dominant | §15.3 | signal de danger → action ou rien | — | `boucle.boucle_temps_reel` (évalué en premier) |

---

## 22. `statistiques.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `residu_normalise` | résidu de Mahalanobis via tête hétéroscédastique $\mathcal N(\mu_\theta,\Sigma_\theta)$ | §4.2 | x, module → résidu | — | `sprt_surprise` |
| `sprt_surprise` | SPRT sur le résidu associé à $\pi_i(x)$ — 1er usage de la statistique | §4.1–4.3 | flux de résidus, $(\alpha,\beta)$ → décision | `utils.sprt_sequentiel`, `residu_normalise` | `memoire_travail.FamilleHorizon.maturer` |
| `sprt_creation` | SPRT sur les échecs de réparation successifs, contextes distincts exigés — 2e usage | M1 | échecs, $(\alpha,\beta)$ → "module manquant" ou non | `utils.sprt_sequentiel` | `graphe.py` (pipeline §4.5, étape 2) |
| `sprt_drift` | SPRT de nouveauté appliqué au domaine d'un module certifié — 3e usage | M10 | résidus récents vs anciens → drift ou non | `utils.sprt_sequentiel` | `graphe.recalage_plancher_drift` |
| `controle_fdr` | Benjamini-Hochberg ou alpha-spending sur l'ensemble des tests de non-infériorité exécutés dans la fenêtre du jour | M10 | liste de p-valeurs/tests → seuil ajusté | — | `graphe.controle_multiplicite` |
| `cadence_variable` | fixe la cadence d'échantillonnage de chaque SPRT selon l'échelle temporelle du flux concerné (rapide pour le sensorimoteur, lente et déclenchée pour une croyance de haut niveau) | §4.4 | type de flux → cadence | — | `sprt_surprise` |

---

## 23. `inne.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `construire_graphe_inne` | graphe de naissance minimal : moteur, réflexe câblé, module visuel par défaut, discriminateur partagé instancié une fois | §1.3 (bootstrap) | — → Graphe | `module_visuel.ModuleVisuel.__init__`, `discriminateur.Discriminateur.__init__` | `boucle.main_loop` |
| `reflexe_frein` | réflexe de survie câblé, jamais appris, jamais atrophié | §15.3 (garde-fou câblé) | vitesse, douleur → commande ou rien | — | `decision_action.reflexe_cable` |

---

## 24. `boucle.py`

| Fonction | Rôle | §SCL | Entrées → Sorties | Appelle | Appelé par |
|---|---|---|---|---|---|
| `construire_contexte_enrichi` | résumé fixe du contexte (composante $\mathcal F_t^{\text{ctx}}$) | §1.2 | contexte brut, besoins, erreur → tenseur | — | `boucle_temps_reel` |
| `inputs_bruts` | capteurs bruts en tenseurs (composante physique de $\mathcal F_t^{\text{ptr}}$) | §1.2 | contexte, besoins → dict | — | `boucle_temps_reel` |
| `boucle_temps_reel` | pas de temps réel complet : construction de $T_t$, décision de composition, exécution, apprentissage local, mise à jour des besoins et de la mémoire de travail | §4.5, §10, §15 | contexte, tables, mémoires, monde, t → commande | `attention.construire_T_t/macro_pas`, `decision_action.priorisation_besoin_dominant`, `graphe.localiser_point_branchement`, `module.entrainer_module_reco/gen` | `main_loop` |
| `reve_coordonne` | entraînement contrastif de la gate par contraste imagination perturbée / monde latent | §8.3 (esprit) | registre_cablage, table_contexte → (mutation) | `discriminateur.entrainer_contrastif` | `cycle_nocturne` |
| `cycle_nocturne` | pipeline nocturne complet : rejeu contrefactuel, SPRT de création, recalage drift, purge de provenance, consolidation FDR, entraînement REINFORCE des pointeurs, entraînement $V_\psi$ | §4.5, §7.2, §8, §9, M1, M6, M10 | graphe, mémoires, t → (mutation globale) | `credit.rejeu_contrefactuel_nocturne`, `statistiques.sprt_creation/drift`, `graphe.controle_multiplicite`, `memoires.RegistreProvenance.purger`, `attention.entrainer_pointeurs`, `recherche.entrainer_v_psi` | `main_loop` |
| `main_loop` | boucle jour/nuit complète, persistance | — | paramètres de run → (graphe, monde, besoins) | `boucle_temps_reel`, `cycle_nocturne`, `checkpoint.sauvegarder/charger` | point d'entrée du programme |

---

## 25. Ordre de construction recommandé

1. **Socle** : `config.py`, `utils.py` (dont `sprt_sequentiel`), `logger.py`, `checkpoint.py`, `monde.py`.
2. **Mémoires de base** : `memoires.py` (besoins, contexte, tampon, exceptions, registres).
3. **Unité d'apprentissage** : `module.py` (avec $\pi_i(x)$, $\bar g_i$, canal réinjecté dès la première version), `module_visuel.py`.
4. **Brique transverse** : `discriminateur.py` — construite une fois, avant tout ce qui la consomme.
5. **Disponibilité et non-infériorité** : `disponibilite.py`, puis `graphe.py` (structure + `test_non_inferiorite` avant les fonctions qui en dépendent).
6. **Statistiques génériques** : `statistiques.py` (dépend de `utils.sprt_sequentiel`).
7. **Pipeline de création** : `simulateur.py`, `graphe.localiser_point_branchement`, `graphe.creer_module_candidat` (création jumelée). Valider en isolation avant intégration à la boucle temps réel.
8. **Mémoire de travail et multi-échelle** : `memoire_travail.py`.
9. **Recherche et crédit** : `recherche.py`, `credit.py` (dépendent du rejeu contrefactuel).
10. **Attention** : `attention.py` — le chantier le plus dense, construit et validé en dernier parmi les mécanismes cognitifs, une fois toutes ses dépendances (§1–§9, §11–§13) disponibles.
11. **Allocation et décision** : `allocation_attention.py`, `decision_action.py`.
12. **Bootstrap et boucle** : `inne.py`, `boucle.py` — assemble tout.
13. **Différé** : `operateurs_natifs.py` (dépend du vocabulaire d'opérateurs stabilisé dans `attention.py`).

---

## 26. Index fonction → section SCL

| Fonction | Fichier | §SCL |
|---|---|---|
| `besoin_dominant` | `memoires.py` | §15.3 |
| `RegistreDisponibilite.*`, `RegistreProvenance.*` | `memoires.py` | §1.4, §8.3 |
| `fiabilite_contextuelle`, `incorporer_gradient` | `module.py` | §1.4, §1.3 |
| `entrainer_masque` | `module_visuel.py` | §1.3 |
| `Discriminateur.*` | `discriminateur.py` | §5 |
| `disponibilite_anticipee`, `logique_acceptation` | `disponibilite.py` | §1.4 |
| `croissance_gouvernee` | `graphe.py` | §2.2 |
| `rejet_gouverne` | `graphe.py` | §2.3 |
| `localiser_point_branchement` | `graphe.py` | §4.6 |
| `test_non_inferiorite`, `consolidation_n_vers_un`, `controle_multiplicite`, `recalage_plancher_drift` | `graphe.py` | §9, M10 |
| `Simulateur.*` | `simulateur.py` | §4.5, §10.2 |
| `construire_T_t` … `accumulateur_orchestrateur` | `attention.py` | §10 |
| `a_etoile` … `a_etoile_ancree` | `recherche.py` | §7 |
| `regret_composition`, `approx_regret_jour`, `rejeu_contrefactuel_nocturne`, `amorcage_creation` | `credit.py` | §10.8, §8.1, §8.2 |
| `allouer_capacite`, `urgence_fil`, `role_creation` | `allocation_attention.py` | §13 |
| `TamponRelatif.*`, `FamilleHorizon.*`, `PalierSommeil.*` | `memoire_travail.py` | §11, §12, §10.7 |
| `primitives_isa`, `garde_fou_domaine`, `composition_sure` | `operateurs_natifs.py` | §14 |
| `fusion_ponderee`, `recompense_intrinseque`, `priorisation_besoin_dominant`, `generer_actions_candidates`, `reflexe_cable` | `decision_action.py` | §15 |
| `residu_normalise`, `sprt_surprise`, `sprt_creation`, `sprt_drift`, `controle_fdr`, `cadence_variable` | `statistiques.py` | §4, M1, M10 |

---

## 27. Orchestrateur — contraintes établies et boîte à outils (maj 2026-07-19)

Section ajoutée d'un commun accord après reprise « étape par étape ». Elle fige
les contraintes retenues (théorie + vérifié empiriquement) et le catalogue
d'outils que l'orchestrateur devra, à terme, savoir créer/choisir/composer. Le
choix parmi ces outils est **naïf** (catalogue + MDL) au départ, **appris par
renforcement** ensuite selon le contexte.

### 27.1 Contraintes

**Apprentissage.** Local, aucun gradient global entre modules (§2). **Parcimonie/
MDL est le moteur** (§5) : un module doit RÉDUIRE ses entrées/sorties (goulot) ;
la taille INTERNE peut être grosse (générateur de qualité), seul le goulot compte.
Un champ discret se reconstruit par **classification par cellule pondérée** (sinon
effondrement à zéro, cf. échec README). Pour des données positionnelles, mettre de
la **convolution AVANT le goulot** (un MLP pur plafonne ~44 %).

**Modules.** Objet générique **détecteur/générateur (E, G)**, GPU, chargeable/
**dormant**, réactivable (§1.3). Portent une **incertitude/fiabilité** = indicateur
de contexte (ex. vitesse : fiabilité 84 % à la vitesse d'entraînement vs ~20 %
ailleurs — vérifié). Verrouillage **asymétrique** (plancher, jamais plafond, §1.4).

**Création & structure.** Déclenchée par **surprise confirmée (SPRT)**, jamais un
calendrier (§4.5). Cascade coût-croissant réparation → composition → création.
**Discriminateur D_φ partagé** (§5). Cycle **jour/nuit** (§8).

**Orchestration.** L'orchestrateur **compose, ne calcule jamais** lui-même (§10.2).
Toute composition **s'ancre à un point vérifiable** (§7.4). **Activation creuse**
(§10.7). Le **choix d'architecture/dimension est une ACTION** de l'orchestrateur
(catalogue + MDL, naïf puis RL — `orchestrateur_naif.essayer_catalogue`, vérifié :
sur [8..96] le MDL choisit dim=48).

**Matériel.** Titan Black (Kepler) : watchdog GPU → **un module entraîné à la fois**.

### 27.2 Boîte à outils cible (types de modules)

| outil | rôle | fichier / état |
|---|---|---|
| **Classification émergente** (VQ) | découvre SANS étiquette les SORTES d'éléments (codebook, catégories utilisées émergent, inutiles élaguées) ; chaque catégorie = module identifie+régénère (« un sucre »). Base de la reconstruction/objets. | `classification_emergente.py` ✅ (4 catégories 100 % pures, reconstruction 100 %) |
| **Compresseur** (E,G) | champ → latent réduit → champ (goulot, classif. pondérée) | `module_ae.py` ✅ (~90 %) |
| **Prédicteur/transition** | latent/champ P-1 → P ; fiabilité = indicateur de vitesse | `module_ae.py` ✅ (84 %) |
| **Attention/masquage** | sélectionne une région/un objet du champ → **plusieurs modules spécialisés** → latent STRUCTURÉ (liste d'objets) → prédiction triviale | ❌ à faire *(slot-attention ; clé vision)* |
| **Mémoire de lieu** | stocke une signature compressée, dormante, **réactivée** au revisit (carte mentale) | ❌ à faire |
| **Délai (T-1, T-2…)** | recopie l'output d'un module au pas précédent (registre/trace), empilable | `composition.py` ✅ |
| **Module de transformation** (« vitesse ») | latent(T-1) → latent(T) ; UN module par RÉGIME ; naît quand un régime nouveau n'est plus expliqué ; se VERROUILLE une fois compétent | `composition.py` ✅ (3/3 régimes détectés) |
| **Discriminateur** D_φ | plausibilité réel/halluciné | `discriminateur.py` ✅ |
| **Simulateur** S_new | mémoire épisodique générative (rejeu nocturne) | `simulateur.py` ✅ |

*(Opérateurs natifs §14 : écartés — non nécessaires pour ce POC.)*

### 27.3 Opérations (les « verbes » de l'orchestrateur)

- **Créer** un module : choisir *(type, dimension, entrée-source)* et sélectionner
  par **MDL** (`essayer_catalogue`). Naïf ✅ pour la dimension ; type/source à ajouter.
- **Composer** : triplets (source, opérateur, cible) via Pointer Network (§10.2 ;
  machinerie présente, pas encore pilote de l'action).
- **Router/masquer** : activation creuse ; **réactiver un module dormant** sur
  correspondance de contexte.
- **Cycle de vie** : verrouiller, fragmenter/découper, consolider (n→1), atrophier,
  endormir/réveiller (§9 ; code présent, pas branché).
- **Évaluer** : MDL, fiabilité, surprise (SPRT), regret de composition.
- **Méta (RL, plus tard)** : apprendre *quel* outil/dimension/composition choisir
  selon le contexte.

### 27.3bis Enseignements empiriques sur la COMPOSITION (étape 6)

Composer `compresseur → délai → module-transformation → générateur` et évaluer aux
DEUX niveaux (latent réel du module, puis champ régénéré vs champ réel) fait bien
ÉMERGER un module par régime, qui détecte la vitesse (3/3 mesuré). Quatre
conditions se sont révélées NÉCESSAIRES — à garder pour tout futur outil :

1. **Créer sur surprise CONFIRMÉE + délai de grâce.** Sans ça, un module naît à
   chaque pas (mesuré : 799), le nouveau-né étant lui aussi non entraîné.
2. **Normaliser le latent** avant de le donner à un module aval : le latent brut
   d'un compresseur a des magnitudes arbitraires (résidus à 50-60, tout seuil
   devient absurde).
3. **Critère de surprise SANS UNITÉ, borné et lissé** : résidu relatif au prior
   trivial « rien ne change », plafonné (la distribution est à queue lourde) et
   lissé (EMA) — un compteur de pas consécutifs ne déclenche jamais.
4. **Verrouillage asymétrique (§1.4) = condition de la spécialisation.** Sans
   verrou, un module compétent se ré-entraîne sur le régime suivant et OUBLIE le
   sien : plus de spécialisation, donc plus de détection possible. C'est le verrou
   qui force la NAISSANCE d'un module pour un régime nouveau. (Confirmation
   empirique du rôle du verrou : ce n'est pas un raffinement, c'est structurant.)

### 27.4 Schéma cible pour la vision

Pas un compresseur global unique (latent opaque → prédiction dure, vérifié : chaîne
1→2→1 à 57 %), mais un enchaînement d'outils GÉNÉRIQUES :
`champ → CLASSIFICATION ÉMERGENTE (VQ : découvre les types d'éléments) →
ATTENTION/MASQUE (groupe les cellules d'un même type non-fond en OBJETS) →
latent STRUCTURÉ (liste d'objets typés) → prédiction triviale (position → position+v)`.

**Dette technique à résorber (codage en dur à retirer)** : l'outil attention actuel
(`module_attention.py`) reconstruit via une **tête de classification à 4 classes
DONNÉES** (`VALEURS`), et la « liste d'objets » lit ce champ classifié — les types
sont donc imposés, pas émergents. À remplacer : l'attention doit se construire sur
les **catégories émergentes** de `classification_emergente.py` (aucune classe
donnée). Principe (Architecture) : on ne programme que du générique ; toute
adaptation manuelle (ici, la tête 4-classes) doit être documentée comme dette et
remplacée par l'outil émergent correspondant.
