Cadre formel {#sec:cadre}
============

Environnement et besoins
------------------------

POMDP $\langle \mathcal{S},\mathcal{A},\mathcal{O},T,\Omega,R\rangle$. État $s_t=(p_t,v_t,\Gamma_t,b_t)$. Action $a_t\in\mathcal{A}$ (accélérations discrètes bornées). Observation $o_t=(V_t,\pi_t)$ (champ visuel, proprioception).

Vecteur de besoins, portée initiale à 2 composantes :
$$b_t = (\text{faim}_t,\ \text{ennui}_t)\in[0,1]\times[0,0.5]$$
$\text{ennui}_t=\min\big(0.5,\ f(t-t_{\text{dernière surprise validée}})\big)$, $f$ croissante, $f(0)=0$ (surprise validée, §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}). Plafond dur à $0.5$ : structurel, pas appris.

**\[D\]** Fatigue, peur, capacité de création : hors périmètre initial (déférées --- le simulateur impose la nuit, pas de source de danger ni d'objectif de production dans le monde actuel). Incertitude résiduelle du monde exploré non traitée comme composante séparée : absorbée dans le sens de $\text{ennui}_t$ (sortir de sa zone prévisible réduit l'incertitude par construction, pas de second signal redondant). Extension à $n>2$ composantes : même vecteur, même mécanisme de sélection (§[15](#sec:action){reference-type="ref" reference="sec:action"}), pas de refonte structurelle attendue.

Catalogue de flux
-----------------

$$\mathcal{F}_t = \mathcal{F}_t^{\text{ctx}} \cup \mathcal{F}_t^{\text{ptr}}$$
**\[D\]** $\mathcal{F}_t^{\text{ctx}}$ : information de conditionnement (condensateurs $c_i$, fiabilités $\pi_i$, besoins $b_t$, état de verrouillage) --- consommée par l'attention (§[10](#sec:attention){reference-type="ref" reference="sec:attention"}) comme contexte, jamais pointable comme source/cible d'un triplet.
$$\mathcal{F}_t^{\text{ptr}} = \{\text{capteurs bruts}\}\cup\{z_i(t): i\in A_t\}\cup\{\text{trace}_{t-1}\}$$
Éléments homogènes en type/dimension, ce sont les éléments pointables de $T_t$ (§[10](#sec:attention){reference-type="ref" reference="sec:attention"}). $A_t$ : modules disponibles (§[1.4](#sec:asym){reference-type="ref" reference="sec:asym"}). $\text{trace}_{t-1}$ : séquence des décisions (source, opérateur, cible) émises au pas précédent (§[10.6](#sec:meta){reference-type="ref" reference="sec:meta"}).

Module
------

$M_i=(E_i,G_i)$, $E_i:\mathbb{R}^{d_i}\to\mathbb{R}^{k_i}$, $G_i:\mathbb{R}^{k_i}\to\mathbb{R}^{d_i'}$, paramètres $\theta_i^E,\theta_i^G$ :
$$\mathcal{L}_i(\theta_i^E,\theta_i^G)=\mathbb{E}_{x\sim\mathcal{D}_i}\big[d(x',G_i(E_i(x)))\big]$$
Condensateur $c_i\in[0,1]$, renforcement (monte sur succès, descend sur échec).

**\[D\]** Sortie légèrement plus large que l'entrée : $d_i' = d_i^{\text{cons}}+d_i^{\text{réinj}}$. La part $d_i^{\text{réinj}}$ alimente $\mathcal{F}_{t+1}^{\text{ptr}}$ directement --- généralisation de la trace auto-référentielle (§[10.6](#sec:meta){reference-type="ref" reference="sec:meta"}) à chaque module, pas seulement à l'orchestrateur. Dimensionnement, Annexe B.

**\[D\]** Discriminateur partagé, pas un par module : $D_\phi$ (§[5](#sec:mecanisme){reference-type="ref" reference="sec:mecanisme"}) sert de validateur de plausibilité pour tout module et pour le rejeu nocturne (§[10.7](#sec:sparse){reference-type="ref" reference="sec:sparse"}, §[8](#sec:historique){reference-type="ref" reference="sec:historique"}). $M_i$ reste $(E_i,G_i)$, deux réseaux, aucune duplication.

**\[D\]** Module par défaut pour l'entrée sensorielle brute (ex. champ visuel) : CNN encodeur-décodeur donné a priori, pas découvert par l'orchestrateur, entraîné par reconstruction masquée auto-supervisée (JEPA --- LeCun, 2022 ; Assran et al., 2023). Dimensionnement proposé, Annexe B.

Exemple de chaînage filé : $G_{\text{vitesse}}$ prédit une vitesse à $t{+}1$ ; chaînée en entrée du générateur visuel $G_{\text{visuel}}$, elle produit un champ visuel prédit à $t{+}1$, validé contre l'observation réelle par $D_\phi$ (§[5](#sec:mecanisme){reference-type="ref" reference="sec:mecanisme"}, §[10.2](#sec:triplet){reference-type="ref" reference="sec:triplet"}).

**\[D\]** Accumulateur de gradient persistant par module, statistique de moment (Polyak, 1964 ; Kingma & Ba, 2014) :
$$\bar g_i \leftarrow \beta\,\bar g_i + (1-\beta)\,\nabla_{\theta_i}\mathcal{L}_i(x,y)$$
Jour : mise à jour à cadence rapide sur lots réels ($\beta$ bas). Nuit : même accumulateur, incorporant des exemples générés (§[8](#sec:historique){reference-type="ref" reference="sec:historique"}.3, §[10.7](#sec:sparse){reference-type="ref" reference="sec:sparse"}), $\beta$ plus élevé (lissage renforcé). Un seul état par module, deux cadences.

Disponibilité anticipée et verrouillage asymétrique {#sec:asym}
---------------------------------------------------

Plateau de progrès : $\rho_i(t)=\text{pente}\big(\{\mathcal{L}_i(x)\}_{x\in\mathcal{W}_i(t)}\big)$ (régression linéaire sur un échantillon $\mathcal{W}_i(t)$ de taille $w$, contextes variés et distincts, pas nécessairement consécutifs). Stabilité du bruit résiduel : $\hat\sigma_i(t)$ (tête hétéroscédastique, §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}), $\text{Var}(\hat\sigma_i)$ sur $\mathcal{W}_i(t)$.

$M_i$ disponible dans $\mathcal{F}_t^{\text{ptr}}$ (avant verrouillage complet) $\iff |\rho_i(t)|<\epsilon_s$ et $\text{Var}(\hat\sigma_i)<\epsilon_\sigma$ sur $\mathcal{W}_i(t)$.

**\[D\]** Critère inspiré de la détection de rupture (CUSUM, Page, 1954), non une application certifiée (régime non stationnaire). Échantillon varié plutôt que fenêtre glissante : une stabilité apparente sur des mesures consécutives peut simplement refléter un contexte immédiat inchangé, pas une robustesse réelle du module.

Verrouillage $=$ plancher certifié $c_i^{\min}$ (meilleure erreur relative atteinte), jamais plafond : toute mise à jour ultérieure de $\theta_i$ est acceptée ssi elle passe le test de non-infériorité (§[9](#sec:rebranch){reference-type="ref" reference="sec:rebranch"}) contre $c_i^{\min}$.

**\[D\]** $\pi_i(x)=1-\hat{\mathcal{L}}_i^{\text{relative}}(x)$ : fiabilité courante indexée par contexte $x$ (et non par instant $t$) --- détecte qu'un module échoue spécifiquement sur telle configuration d'entrée, indépendamment de sa moyenne globale. Consommée par $V_\psi$ (§[7](#sec:search){reference-type="ref" reference="sec:search"}) et par le gating (§[10](#sec:attention){reference-type="ref" reference="sec:attention"}).

**\[D\]** Logique d'acceptation d'un nouvel exemple $(x,y)$ : incorporation à $\bar g_i$ (§1.3). Si $\pi_i(x)$ augmente après incorporation, mise à jour validée par le test de non-infériorité (§[9](#sec:rebranch){reference-type="ref" reference="sec:rebranch"}, inchangé). Si $\pi_i(x)$ diminue, $\theta_i$ n'est pas mis à jour sur cet exemple ; à la place, le prédicteur de $\pi_i$ est entraîné à annoter ce contexte comme peu fiable pour $M_i$ --- signal consommé par le pipeline de réparation/recherche (§[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.5).

**\[D\]** Statut provisoire pour les modules nés du pipeline de création (§[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.5, §[10.2](#sec:triplet){reference-type="ref" reference="sec:triplet"}) : jamais de plancher $c_i^{\min}$ verrouillé sur la seule base du rejeu simulé via $S_{\text{new}}$. Le test de non-infériorité (§[9](#sec:rebranch){reference-type="ref" reference="sec:rebranch"}) mesuré contre des exemples issus de $S_{\text{new}}$ ne compte pas pour la certification ; seule compte la non-infériorité contre une occurrence réelle ultérieure du même contexte. Sans confirmation réelle : statut provisoire, pas de plancher, éligible à la purge (§[8](#sec:historique){reference-type="ref" reference="sec:historique"}.3). Sinon, le rejeu nocturne consoliderait du sur-apprentissage sur variations d'un point unique --- au pire une hallucination avec statut de mémoire réelle.

Apprentissage local sans gradient global
========================================

Descente par blocs
------------------

$\Phi(\theta)=\sum_i w_i\mathcal{L}_i(\theta_i^E,\theta_i^G)$ ; mise à jour de $\theta_i$ seul $=$ Gauss-Seidel par blocs.

**\[P\]** Sous $L$-lissité, pas $\le 1/L$ (Tseng, 1993 ; Bottou, Curtis & Nocedal, 2018) :
$$\mathbb{E}[\mathcal{L}_i(\theta_i-\eta\nabla\mathcal{L}_i)]\le\mathcal{L}_i(\theta_i)-\frac{\eta}{2}\|\nabla\mathcal{L}_i(\theta_i)\|^2+O(\eta^2\sigma^2)$$
Pas de garantie de convergence globale (non-convexité par bloc).

**\[D\]** Remarque de portée : ce document retient l'apprentissage strictement local (aucun gradient global) par défaut. Alternative gradient global explicitement flaguée comme piste expérimentale à tester au moment du code, non tranchée ici.

Croissance structurelle gouvernée {#sec:monotonie}
---------------------------------

$$\Phi_t=\sum_{i\in A_t}\hat{\mathcal{L}}_i^{\text{relative}}(t)$$
$\sigma$ acceptée $\iff \Phi_t(\text{après})<\Phi_t(\text{avant})-\varepsilon$.

À $A_t$ fixé (entre deux événements structurels : création, fragmentation, consolidation), $(\Phi_t)_t$ est non croissante sous acceptation $\varepsilon$-stricte, bornée inférieurement --- monotonie par morceaux.

**\[D\]** La convergence globale n'est pas garantie par cette proposition : $A_t$ change de cardinal à chaque événement structurel (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.4, §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.5), ce qui rompt la monotonie d'une somme à support variable. Reclassé de \[P\] global à \[P\] local + \[D\] pour l'articulation entre morceaux --- aucune borne globale prouvée sur le nombre d'événements structurels ni sur leur effet net cumulé. Limite reportée au tableau §16.

Correspondance code : déclencheur de croissance structurelle sur les objets $(E_i,G_i,\pi_i,c_i)$ de §[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.3 --- fonctions `decouper_module`/`fragmenter_module` de l'implémentation existante.

Rejet gouverné par gating conditionnel
--------------------------------------

$g_i(x)=\sigma(w_i^\top x+b_i)$, testé contre $H_0$ (contexte non informatif) par rapport de vraisemblance $G^2=-2\log\frac{\mathcal{L}(H_0)}{\mathcal{L}(H_1)}$.

**\[P\]** $G^2\sim\chi^2_{df}$ sous $H_0$ (Wilks, 1938), asymptotique. $M_i$ conservé si $G^2$ dépasse le quantile critique ; sinon rejeté (désuétude).

Correspondance code : déclencheur de gating/élagage de l'orchestrateur, consommant $\pi_i(x)$ (§[1.4](#sec:asym){reference-type="ref" reference="sec:asym"}) comme feature d'entrée du test.

Compression
===========

Rate-distortion
---------------

$$R(D)=\min_{p(z|x):\mathbb{E}[d(X,\hat X)]\le D} I(X;Z)$$
Un module réalise un majorant heuristique de $R(D)$ à $k=\dim(Z)$ fixé.

Multi-fidélité
--------------

**\[P\]** Aucune unicité du point de fonctionnement (Shannon, 1948, 1959) : $\{E^{(l)}\}_{l=1}^m$ à distorsions $D^{(1)}<\dots<D^{(m)}$ coexistent, chacun servant un consommateur distinct.

MDL et limite gloutonne
-----------------------

$$L_{\text{total}}(k)=\underbrace{L(\theta_k)}_{\propto k}+\underbrace{L(\mathcal{D}\mid\theta_k)}_{\propto n\hat{\mathcal{L}}_i(\theta_k)}$$
**\[D\]** Croissance gloutonne ($k\to k+1$ seulement si gain immédiat) non garantie optimale sur paysage non monotone (Frankle & Carbin, 2018 ; Hinton, Vinyals & Dean, 2015). Alternative de sur-paramétrisation temporaire non tranchée.

Détection de nouveauté {#sec:surprise}
======================

**\[D\]** Détection non temporelle : toute zone de contexte où $\pi_i(x)$ (§[1.4](#sec:asym){reference-type="ref" reference="sec:asym"}) est bas, indépendamment de l'instant où $x$ est rencontré. Les tests ci-dessous opèrent sur le résidu associé à $\pi_i(x)$, pas sur un flux temporel brut : $\pi_i(x)$ bas $\iff$ résidu normalisé $S(x)$ élevé (§[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.2), même quantité, deux angles.

Rapport de vraisemblance
------------------------

$$S(x_{t+1})=\log\frac{p_0(x_{t+1}\mid x_t)}{p_\theta(x_{t+1}\mid x_t,\text{ctx})}$$

$S(x)>\tau$ est le test le plus puissant à $\alpha$ fixé, pour $H_0,H_1$ simples.

Portée : valide si $H_0,H_1$ correctement spécifiées ; $H_1$ inconnue a priori ici.

Résidu normalisé
----------------

Tête hétéroscédastique $\mathcal{N}(\mu_\theta(z),\Sigma_\theta(z))$ ; $S(x)=(x-\mu_\theta)^\top\Sigma_\theta^{-1}(x-\mu_\theta)\sim\chi^2_d$ sous $H_0$.

Récurrence : SPRT
-----------------

$$\Lambda_n=\sum_{i=1}^n\log\frac{p_1(S(x_i))}{p_0(S(x_i))}$$

Le SPRT minimise le nombre moyen d'observations parmi tous les tests à $(\alpha,\beta)$ fixés.

Cadence variable
----------------

**\[D\]** La cadence d'échantillonnage de $\Lambda_n$ n'est pas fixe : rapide (chaque pas) pour un flux sensorimoteur, lente et déclenchée par pertinence contextuelle pour une croyance de haut niveau (§[12](#sec:multiechelle){reference-type="ref" reference="sec:multiechelle"}). Même statistique, cadence différente selon l'échelle temporelle du flux concerné.

Pipeline de réparation et de recherche
--------------------------------------

**\[D\]** Séquence déclenchée quand $\pi_i(x)$ franchit le seuil bas (test §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.1--[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.3) :

1.  Réparation locale : $(x,y)$ incorporé à $\bar g_i$ (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.3), mise à jour locale (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.4, BCD §2.1). Si $\pi_i(x)$ repasse au-dessus du seuil, arrêt.

2.  **\[NOUVEAU\]** Échecs de réparation accumulés comme observations d'un SPRT de création (réemploi §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.3, troisième usage de la même statistique) : $\Lambda_n$ sur les échecs successifs, contextes distincts exigés (échantillon varié, réemploi §[1.4](#sec:asym){reference-type="ref" reference="sec:asym"}). Franchissement du seuil $\Rightarrow$ conclusion « module manquant », pas « module perfectible ».

3.  Test de plausibilité par $D_\phi$ (§[5](#sec:mecanisme){reference-type="ref" reference="sec:mecanisme"}, mécanisme inchangé) : la chaîne de modules disponible (§[10.2](#sec:triplet){reference-type="ref" reference="sec:triplet"}) peut-elle générer un simulacre de $x$ jugé plausible ? Simulacre plausible $\Rightarrow$ recherche A$^*$ (§[7](#sec:search){reference-type="ref" reference="sec:search"}) d'insertion/branchement. Simulacre implausible **et** SPRT de création franchi (étape 2) $\Rightarrow$ étape 5.

4.  Simulacre plausible $\Rightarrow$ mise en file nocturne (§[7](#sec:search){reference-type="ref" reference="sec:search"}, §[8](#sec:historique){reference-type="ref" reference="sec:historique"}). Module(s) candidat(s) conservé(s) ssi le test de non-infériorité (§[9](#sec:rebranch){reference-type="ref" reference="sec:rebranch"}) est franchi après recherche.

5.  **\[NOUVEAU\]** Simulacre implausible et SPRT de création franchi $\Rightarrow$ création jumelée (§[10.2](#sec:triplet){reference-type="ref" reference="sec:triplet"}), sous budget WFQ (§[13](#sec:allocation){reference-type="ref" reference="sec:allocation"}).

**\[D\]** Sans le SPRT de l'étape 2, le système oscille entre réparer indéfiniment un module inadapté et proliférer un module par simple fluctuation. $(\alpha,\beta)$ de ce SPRT, Annexe B.

Correspondance code : suivi de $\pi_i$ (`evaluer_previsions`), point d'entrée de recherche (`chercher_latent_predictif`/`aligner_action`), sélection de chantier nocturne (`pilote.prioriser()`).

Règle de localisation du point de branchement {#sec:localisation}
---------------------------------------------

**\[D\]** Point de branchement = premier module au sens du flux dont $\pi_i(x)$ s'effondre alors que tous ses antécédents directs conservent $\pi$ haut sur ce même contexte $x$. Balayage de $\pi$ le long de la chaîne active au moment de l'échec --- les $\pi_i(x)$ existent déjà (§[1.4](#sec:asym){reference-type="ref" reference="sec:asym"}), seule l'orchestration du balayage est nouvelle. Si le premier $\pi$ effondré est le capteur lui-même : branchement en tête, apprentissage d'un nouveau compresseur d'entrée.

Mécanisme rapide et atténuation douce {#sec:mecanisme}
=====================================

$$\phi^*=\arg\min_\phi -\log D_\phi(x^+)-\sum_{j=1}^N\log(1-D_\phi(x_j^-)),\quad x_j^-\sim p_\theta(\cdot)$$
**\[P\]** Classification generative-contrastive (NCE, Gutmann & Hyvärinen, 2010 ; few-shot bayésien, Lake, Salakhutdinov & Tenenbaum, 2015). $D_\phi$ est réutilisé sans modification comme validateur de plausibilité pour le chaînage de générateurs (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.3) et pour le rejeu nocturne (§[10.7](#sec:sparse){reference-type="ref" reference="sec:sparse"}).

Atténuation : $w_j=\exp(-\lambda r_j)$, $r_j$ rang/magnitude de l'effet attribué à la dimension $j$ --- jamais de masque à zéro (shrinkage, James & Stein, 1961 ; Tibshirani, 1996).

Coût de branchement
===================

$$\mathcal{L}_{\text{réel}}\le\mathcal{L}_{\text{empirique}}+O\!\left(\sqrt{\tfrac{\text{complexité}(E,G)}{n}}\right)$$
(Vapnik, 1998 ; Bartlett & Mendelson, 2002.)

Recherche de composition {#sec:search}
========================

A$^*$
-----

$f(n)=g(n)+h(n)$, optimal ssi $h$ admissible et consistante (Hart, Nilsson & Raphael, 1968).

Heuristique apprise
-------------------

$$\psi\leftarrow\psi+\alpha_t\big(r_n+\gamma V_\psi(n')-V_\psi(n)\big)\nabla_\psi V_\psi(n)$$
$V_\psi$ non garantie admissible : best-first pondéré (Pohl, 1970). Convergence p.s. sous Robbins-Monro (1951) en tabulaire/linéaire (Sutton, 1988 ; Tsitsiklis, 1994) ; non garantie en non-linéaire (Sutton & Barto, 2018). $\pi_i(t)$ (§[1.4](#sec:asym){reference-type="ref" reference="sec:asym"}) est une feature principale de $V_\psi$.

Dégénérescence exhaustive
-------------------------

Historique vide $\Rightarrow V_\psi\equiv0\Rightarrow f(n)=g(n)$ : recherche non informée, même algorithme.

Ancrage de la composition à un point de vérité vérifiable {#sec:ancrage}
---------------------------------------------------------

**\[D\]** Le monde ne renvoie que des entrées physiques brutes (champ visuel, proprioception), jamais un latent abstrait attendu. Toute composition choisie doit être poussée, via la cascade de générateurs (§[10.2](#sec:triplet){reference-type="ref" reference="sec:triplet"}), jusqu'à un point de comparaison vérifiable --- sans quoi aucun résidu n'est mesurable et la composition ne reçoit aucun signal d'évaluation. Chaîne : contexte observé $\to$ branchement choisi (sorties latentes) $\to$ modules de prévision (état futur en latent) $\to$ modules de génération en cascade $\to$ \[arrêt anticipé possible\] $\to$ prédiction d'entrées physiques $\to$ comparaison au réel ou à un module intermédiaire certifié.

**\[D\]** Critère d'arrêt anticipé : la cascade ne descend au niveau brut que si aucun module intermédiaire n'est déjà réputé reproduire fidèlement ce niveau. Test de non-infériorité (§[9](#sec:rebranch){reference-type="ref" reference="sec:rebranch"}, mécanisme inchangé) appliqué à : la sortie du module intermédiaire est-elle non-inférieure à une descente complète jusqu'au brut, pour ce contexte ? Si oui, arrêt à ce niveau suffit. Sinon, descente forcée. Décision évaluée à chaque tentative, jamais fixée a priori.

Deux cibles de vérité, jamais confondues : (A) le réel brut, quand la prédiction est descendue jusque-là ; (B) la sortie d'un module intermédiaire déjà certifié (plancher verrouillé, §[1.4](#sec:asym){reference-type="ref" reference="sec:asym"}), quand suffisant. Le choix entre A et B détermine où le résidu est mesuré ; le mécanisme de mesure (résidu normalisé, tête hétéroscédastique, §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.2) reste inchangé quel que soit le niveau.

Recherche A$^*$ ancrée
----------------------

**\[D\]** À chaque nœud exploré : (1) branchement de modules choisi sur le contexte observé ou déjà simulé selon la profondeur ; (2) sorties produites au niveau abstrait ; (3) poussées en cascade jusqu'au niveau de comparaison déterminé par §[7.4](#sec:ancrage){reference-type="ref" reference="sec:ancrage"} ; (4) prédiction terminale utilisée pour évaluer $g(n)$ (coût simulé accumulé) et $h(n)$ (coût restant appris, §[7](#sec:search){reference-type="ref" reference="sec:search"}.2) ; (5) comparaison entre branches, sélection de la meilleure ; (6) répétition à la profondeur suivante tant que le budget de temps le permet.

Fusion réel/imaginé par profondeur : nœuds proches de la racine mélangent perception réelle et projection à un pas (§[15](#sec:action){reference-type="ref" reference="sec:action"}.1) ; nœuds profonds tournent entièrement sur l'imaginé, confiance décroissante avec la profondeur (propagation d'incertitude, §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.2, inchangée).

Génération de l'historique d'apprentissage {#sec:historique}
==========================================

Rejeu contrefactuel
-------------------

Chaque nuit, pour un échantillon de contextes $x_j$ : rejeu (poids figés) de tous les candidats disponibles contre la cible réelle $y_j$ :
$$\big\{(x_j,i,\hat{\mathcal{L}}_i^{\text{relative}}(x_j,y_j))\big\}_{i\in\text{candidats}(x_j)}$$
**\[P\]** Évaluation hors politique par modèle appris (Dyna, Sutton, 1990 ; bandits contextuels, Li, Chu, Langford & Schapire, 2010 ; Dudík et al., 2011). Entraînement par perte de rang (Burges et al., 2005). Limite : fidélité bornée par les générateurs figés utilisés pour le rejeu.

Amorçage à la création
----------------------

**\[D\]** À l'instant de création d'un module $M_i$ dans le contexte $x_{\text{création}}$, injecter immédiatement $(x_{\text{création}},M_i,\text{positif})$ dans le jeu d'apprentissage du gating (§[10](#sec:attention){reference-type="ref" reference="sec:attention"}) --- amorçage explicite, pas d'attente du rejeu nocturne passif pour la première association.

Augmentation générative par contraste positif/négatif
-----------------------------------------------------

Sur un succès réel (chemin $\gamma^+$ menant à un gain mesuré) : générer, par rollout à travers les générateurs figés, un ou plusieurs chemins contrefactuels $\gamma^-$ non empruntés, en estimer l'issue (imaginée). Entraîner le gating par renforcement sur la paire $(\gamma^+,\text{positif}),(\gamma^-,\text{imaginé négatif})$ à partir d'un seul événement réel.

**\[P\]** Extension du rejeu contrefactuel (§[8](#sec:historique){reference-type="ref" reference="sec:historique"}.1) au cas où l'alternative n'a pas été réellement essayée : l'alternative est simulée, pas observée. Limite : la fidélité de $\gamma^-$ dépend entièrement des générateurs figés --- un chemin négatif imaginé et un chemin négatif réellement vécu sont stockés dans le même format de mémoire (§[11](#sec:memoire){reference-type="ref" reference="sec:memoire"}), donc indiscernables a posteriori dans le jeu d'entraînement du gating.

**\[D\]** Garde-fou contre cette indiscernabilité, aggravée quand tout le jeu initial d'un module créé est imaginé (§[10.2](#sec:triplet){reference-type="ref" reference="sec:triplet"}) : (a) bit de provenance $\{\text{réel},\text{imaginé}\}$ attaché à chaque exemple stocké (coût négligeable, un bit par entrée) ; (b) plafond de ratio imaginé/réel dans tout lot d'entraînement du gating (Annexe B) ; (c) purge --- si la confirmation réelle (§[1.4](#sec:asym){reference-type="ref" reference="sec:asym"}) ne vient jamais pour un module provisoire, le rejet par désuétude (§2.3, mécanisme inchangé) élimine le module *et tous les exemples portant sa provenance* --- la désuétude devient ainsi le mécanisme d'oubli des fausses croyances, sans mécanisme nouveau.

Rebranchement et consolidation {#sec:rebranch}
==============================

$$H_0:\mu_B-\mu_A\le-\Delta \quad\text{contre}\quad H_1:\mu_B-\mu_A>-\Delta$$
Rejeter $H_0$ si la borne sup. de l'IC à $1-\alpha$ sur $\hat\mu_B-\hat\mu_A$ dépasse $-\Delta$ (Blackwelder, 1982), sur données appariées.

Consolidation à plusieurs-vers-un : $n$ modules satellites séparés remplacés par 1 module partagé si $L_{\text{total}}(\text{partagé})<L_{\text{total}}(\text{séparés})$ --- même test, même critère MDL (§3.3), appliqué à une comparaison $n\to1$.

**\[D\]** Exception : la non-infériorité mesurée contre un simulateur $S_{\text{new}}$ nouvellement créé (§[10.2](#sec:triplet){reference-type="ref" reference="sec:triplet"}) ne vaut pas certification --- voir statut provisoire (§[1.4](#sec:asym){reference-type="ref" reference="sec:asym"}).

**\[D\]** Contrôle de multiplicité : des dizaines de tests de non-infériorité par jour exigent une correction --- contrôle du FDR (Benjamini & Hochberg, 1995) ou budget $\alpha$ journalier (alpha-spending) sur l'ensemble des tests exécutés dans la fenêtre.

**\[D\]** Re-calage sous drift : si le SPRT de nouveauté (§[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.3) conclut à un drift durable sur le domaine d'un module certifié, son plancher $c_i^{\min}$ est re-mesuré sur la nouvelle distribution --- le verrouillage protège contre la régression à monde constant, pas contre le monde qui change.

Attention modulaire : forme, sortie, langage {#sec:attention}
============================================

Entrée : un ensemble
--------------------

$$T_t=\big\{(\omega_i,e_i,\tau_i)\big\}_{i\in\mathcal{F}_t}$$
$e_i\in\mathbb{R}^{\dim_{\text{emb}}}$ (contenu), $\omega_i\in\mathbb{R}^{\dim_{\text{op}}}$ (opérateur), $\tau_i$ (type, §[10.5](#sec:transfer){reference-type="ref" reference="sec:transfer"}).

**\[P\]** Invariance par permutation garantie par construction (Set Transformer, Lee et al., 2019).

**\[D\]** Chaque élément de $T_t$ provient d'un module déjà existant --- jamais une valeur brute non compressée si un module l'a déjà résumée ($\mathcal{F}_t^{\text{ptr}}$, §[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.2). Contenu concret : dernier latent de chaque module actif (réel si observé, prédit si encore dans la fenêtre de prévision non révélée, §[12](#sec:multiechelle){reference-type="ref" reference="sec:multiechelle"}), $\text{trace}_{t-1}$ (§[10.6](#sec:meta){reference-type="ref" reference="sec:meta"}), fiabilité $\pi_i(x)$ (§[1.4](#sec:asym){reference-type="ref" reference="sec:asym"}) par élément.

**\[D\]** La profondeur de prévision atteignable est une conséquence de la qualité de compression des modules disponibles (§3), jamais une capacité propre à l'orchestrateur : plus les modules compressent fidèlement une portion du monde, plus $T_t$ contient d'éléments compacts exploitables loin dans le temps.

Sortie : triplet (source, opérateur, cible) par pointeurs {#sec:triplet}
---------------------------------------------------------

Une décision de branchement est un triplet $(\text{src},\text{op},\text{cib})$ où chaque composante est un pointeur --- une distribution catégorielle sur les indices courants de $T_t$ (*Pointer Network*, Vinyals, Fortunato & Jaitly, 2015), pas un symbole d'un vocabulaire fixe :
$$p(\text{src}=j\mid T_t)=\text{softmax}_j\big(u^\top\tanh(W_1 e_j+W_2 q)\big)$$
où $q$ est l'état de décodage courant. Idem pour op, cib. L'ensemble $T_t$ inclut un jeton sentinelle $[\text{NEW}]$ (créer un module) et un opérateur sentinelle $\mathrm{id}$ (§[10.3](#sec:parallele){reference-type="ref" reference="sec:parallele"}).

**\[P\]** Les Pointer Networks résolvent exactement le problème d'un vocabulaire de sortie de taille variable (Vinyals et al., 2015) --- ici, le catalogue $\mathcal{F}_t$ change de taille à chaque module créé/retiré.

Application : $\text{op}$ pointé, appelé sur $\text{src}$ pointé (sens bottom-up ou top-down selon le type d'appel), résultat écrit à l'emplacement $\text{cib}$ pointé (module existant, entrée du pool $T_{t+1}$, ou port de sortie moteur).

**\[D\]** L'orchestrateur ne calcule jamais lui-même une valeur du monde : il compose des appels à des modules, qui eux calculent. Répété en autorégressif via le canal réinjecté (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.3) : chaque triplet émis alimente $T_{t+1}$, permettant d'enchaîner plusieurs décisions à la suite.

**\[D\]** Apprentissage des paramètres du pointeur ($W_1,W_2,u$) : gradient de politique (REINFORCE, Williams, 1992) sur la trajectoire de triplets émise, avec pour baseline le regret de composition (§[10.8](#sec:credit){reference-type="ref" reference="sec:credit"}, réduction de variance par le rejeu contrefactuel, §[8](#sec:historique){reference-type="ref" reference="sec:historique"}.1). **\[H\]** Variance du gradient de politique non garantie faible en non-linéaire ; atténuée par la baseline et par la densité des points d'ancrage (§[7.4](#sec:ancrage){reference-type="ref" reference="sec:ancrage"}, un signal par nœud plutôt qu'en fin de trajectoire seule). Limite reportée au tableau §16.

**\[D\]** Masque de compatibilité de type sur le softmax du pointeur, dérivé de $\tau$ (§[10.5](#sec:transfer){reference-type="ref" reference="sec:transfer"}) : les indices dont le type est incompatible avec l'opérateur pointé reçoivent $-\infty$ avant normalisation --- évite d'apprendre par échec ce qui est connu par construction.

**\[D\]** Création jumelée : $[\text{NEW}]$ instancie deux objets simultanément au point de branchement localisé (§[4.6](#sec:localisation){reference-type="ref" reference="sec:localisation"}) : le module $M_{\text{new}}=(E,G)$ standard (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.3), initialisé sur l'exemple réel $x_{\text{création}}$ ; et un simulateur $S_{\text{new}}$ chargé de refabriquer l'entrée du point de défaillance (brute de capteurs ou sortie du module amont certifié) --- mémoire épisodique générative : on ne stocke pas l'épisode, on stocke une machine capable de le refabriquer, ce qui rend le rejeu nocturne possible à partir d'un seul exemple (cohérence CLS, §[11](#sec:memoire){reference-type="ref" reference="sec:memoire"}).

Contraintes sur $S_{\text{new}}$ : (a) tête hétéroscédastique obligatoire (réemploi §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.2) --- un simulateur sans incertitude conditionnelle ne rejoue pas ; sans elle, la reconstruction ponctuelle d'un exemple unique est une mémorisation triviale qui passe $D_\phi$ sans généraliser. (b) Verdict $D_\phi$ étiqueté **\[H\]** dans ce cas d'usage : $D_\phi$ a été entraîné sur l'ancienne distribution ; face au radicalement neuf, sa plausibilité est elle-même hors distribution. Limite reportée au tableau §16.

Exécution parallèle et synchronisation {#sec:parallele}
--------------------------------------

Un macro-pas produit un lot de $w\le W$ triplets simultanés, $W$ borné par la capacité matérielle (mémoire, largeur de calcul).

**\[D\]** Élément dont la dépendance n'est pas prête au macro-pas courant : $\text{op}=\mathrm{id}$ (opérateur identité, sans paramètre, toujours disponible) --- laisse l'élément inchangé jusqu'au macro-pas suivant.

**\[P\]** Modèle dataflow (Dennis, 1974) : un nœud s'exécute quand ses entrées sont prêtes, sinon reste inerte (bulle de pipeline).

$\mathrm{id}$ sert une seconde fonction : maintenir un contenu dans un emplacement de mémoire de travail sur plusieurs pas sans le modifier (§[11](#sec:memoire){reference-type="ref" reference="sec:memoire"}) --- même primitive, deux usages, pas deux mécanismes.

Critères d'arrêt d'un fil de décodage
-------------------------------------

Un fil s'arrête quand : (a) l'incertitude propagée (Mahalanobis cumulé, §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.2) dépasse un seuil ; (b) aucun triplet candidat ne dépasse le seuil de valeur (§[7](#sec:search){reference-type="ref" reference="sec:search"}) ; (c) $\text{cib}$ atteint un port terminal (sortie motrice ou consommateur hors graphe) ; (d) profondeur maximale absolue (garde-fou, jamais le critère principal). Chaque fil s'arrête indépendamment des autres.

Abstraction inter-dimensionnelle {#sec:transfer}
--------------------------------

**\[D\]** Typage minimal $\tau\in\{\text{spatial-}x,\text{spatial-}y,\text{temporel},\text{intensité},\text{catégoriel}\}$, révisable.

Transfert d'un opérateur $\omega_i$ appris sur (spatial-$x$, spatial-$x'$) vers (temporel-$t$, temporel-$t'$) : autorisé si les types partagent une propriété structurelle déclarée (ordre métrique).

**\[H\]** Suppose une structure analogique linéaire de $\Omega$ ($\omega(\text{spatial})\approx\omega(\text{temporel})$ pour une même opération) --- observée ailleurs (Mikolov et al., 2013 ; Gentner, 1983, structure-mapping), non garantie ici. Testable par ablation (opérateur entraîné sur un seul type, évalué zero-shot sur un autre).

Trace auto-référentielle {#sec:meta}
------------------------

$\text{trace}_{t-1}$ (la séquence de triplets émise au pas précédent) est un élément de $\mathcal{F}_t^{\text{ptr}}$ (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.2) au même titre qu'un capteur ou un latent. **\[P\]** Application récursive de la définition de $T_t$ (§[10](#sec:attention){reference-type="ref" reference="sec:attention"}.1) à sa propre sortie passée --- aucun mécanisme supplémentaire. Généralisée par $d_i^{\text{réinj}}$ (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.3) à chaque module, pas seulement à l'orchestrateur.

Activation creuse et rejeu nocturne {#sec:sparse}
-----------------------------------

**\[P\]** À chaque macro-pas, $w\le W$ modules actifs parmi $|A_t|\gg W$ disponibles (§[10.3](#sec:parallele){reference-type="ref" reference="sec:parallele"}) : activation creuse sur catalogue surcomplet, cf. codage parcimonieux sur dictionnaire surcomplet (Olshausen & Field, 1996) --- peu d'atomes actifs pour un dictionnaire disponible très large.

**\[D\]** Palier mémoire supplémentaire, après les deux de §[11](#sec:memoire){reference-type="ref" reference="sec:memoire"}.1 : modules ou contextes en sommeil, non résidents en mémoire de calcul rapide (stockage hôte lent), maintenus via le même opérateur idle/NOP (§[10.3](#sec:parallele){reference-type="ref" reference="sec:parallele"}) --- pas de mécanisme distinct. Un contexte identifié en journée comme non résolu (étape 4, §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.5) y reste jusqu'à récupération nocturne.

**\[D\]** Précondition avant entraînement nocturne sur un contexte stocké $z_{\text{stocké}}$ : reconstruction $\hat x=G_{\text{chaîne}}(z_{\text{stocké}})$ jugée suffisante par $D_\phi$ (§[5](#sec:mecanisme){reference-type="ref" reference="sec:mecanisme"}) --- potentiellement purement latente, sans régénération de champ visuel si non pertinente au contexte mémorisé.

Accumulateur de gradient de l'orchestrateur {#sec:credit}
-------------------------------------------

**\[D\]** Quand le réel atteint l'échéance d'une prédiction (§[12](#sec:multiechelle){reference-type="ref" reference="sec:multiechelle"}), l'écart alimente deux accumulateurs distincts, jamais confondus : (1) le module concret dont la sortie était prédite --- son propre $\bar g_i$ (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.3), inchangé ; (2) l'orchestrateur lui-même --- un accumulateur $\bar g_{\text{orch}}$, même recette de moment (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.3 ; Polyak, 1964 ; Kingma & Ba, 2014), alimenté par le résidu de pertinence de la composition choisie (la sortie était-elle correcte compte tenu du branchement retenu, indépendamment de la qualité propre du module appelé).

**\[D\]** Décomposition explicite du crédit, réemploi du rejeu contrefactuel (§[8](#sec:historique){reference-type="ref" reference="sec:historique"}.1) comme baseline :
$$\text{résidu module}=\hat{\mathcal{L}}_i(x,y)\quad\text{(module effectivement appelé, alimente }\bar g_i\text{)}$$
$$\text{résidu orchestrateur}=\hat{\mathcal{L}}_{\text{choisi}}(x,y)-\min_{j\in\text{candidats}(x)}\hat{\mathcal{L}}_j(x,y)\quad\text{(regret de composition, alimente }\bar g_{\text{orch}}\text{)}$$
Module médiocre bien choisi $\Rightarrow$ regret $\approx0$, résidu module élevé. Bon module mal branché $\Rightarrow$ regret élevé, résidu module bas --- les deux fautes séparées par une quantité déjà calculée chaque nuit par le rejeu contrefactuel. Le jour, approximation du regret par $V_\psi$ des alternatives non prises (§[7](#sec:search){reference-type="ref" reference="sec:search"}.2, réemploi).

Cadence : $\bar g_{\text{orch}}$ s'accumule en continu pendant la journée ($\beta$ bas), se consolide la nuit ($\beta$ plus élevé, incorporant les exemples générés par rejeu contrefactuel, §[8](#sec:historique){reference-type="ref" reference="sec:historique"}.1, §[8](#sec:historique){reference-type="ref" reference="sec:historique"}.3). Seule la consolidation profonde reste nocturne ; l'accumulation elle-même est continue.

Deux systèmes de mémoire {#sec:memoire}
========================

Mémoire de travail : ligne à retard à décalage relatif
------------------------------------------------------

La mémoire de travail est un tampon circulaire de taille fixe $2K+1$, indexé par offset relatif $\delta\in\{-K,\dots,K\}$ par rapport à l'instant courant --- jamais par horaire absolu. À chaque tick, le tampon décale : le contenu à l'offset $\delta$ devient celui à l'offset $\delta-1$ ; une nouvelle observation entre à $\delta=0$ ; le contenu sortant à $\delta=-K$ est archivé ou perdu.

**\[P\]** Ligne à retard (*tapped delay line*), objet standard du traitement du signal (filtre RIF --- cf. Oppenheim & Schafer, *Discrete-Time Signal Processing*). Un emplacement maintenu par $\mathrm{id}$ (§[10.3](#sec:parallele){reference-type="ref" reference="sec:parallele"}) reste inchangé jusqu'à décision explicite de l'orchestrateur de le libérer.

Hiérarchie à deux vitesses : un nombre restreint d'emplacements rapides (« têtes », borné par $W$) ; au-delà, un tampon plus lent et plus grand (registre vs. mémoire principale, analogie directe avec la hiérarchie mémoire d'un calculateur).

Mémoire de certitude (paramétrique)
-----------------------------------

Poids $\psi$ ($V_\psi$), $W_{\text{gate}}$, embeddings $(e_i,\omega_i)$ --- mémoire distincte, non indexée par le temps, mise à jour lente (nocturne/renforcement), jamais lue comme un tampon d'activations.

**\[P\]** Distinction Complementary Learning Systems (McClelland, McNaughton & O'Reilly, 1995) : mémoire rapide indexée par contexte (ici, tampon de travail) vs. mémoire lente, généralisante, intégrée dans les poids (ici, $V_\psi,W_{\text{gate}}$).

Prédiction multi-échelle {#sec:multiechelle}
========================

Généralisation de §[11](#sec:memoire){reference-type="ref" reference="sec:memoire"}.1 : plusieurs familles d'horizon concurrentes $h\in\mathcal{H}=\{h^{(1)},h^{(2)},\dots\}$ (courte, moyenne, longue portée), chacune avec sa propre cadence d'émission et son propre tampon à décalage relatif.

Émission : à cadence propre à la famille, une prédiction est produite à l'offset relatif $+h^{(k)}$ de son tampon. Maturation : quand l'offset relatif atteint $0$ dans ce même tampon (après $h^{(k)}$ décalages depuis son émission), comparaison au réel par le test de §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"} avec la cadence de §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.4 propre à cette échelle.

**\[D\]** Aucune indexation par horaire absolu : chaque prédiction porte son propre offset relatif à l'émission, décrémenté à chaque tick --- le glissement (une nouvelle prédiction à $+h$ émise à chaque tick, maturant $h$ pas après sa propre émission) est une propriété structurelle du tampon, non une logique ajoutée.

Borne de mémoire : nombre d'entrées en vol $\le\sum_k W_k\cdot h^{(k)}/\text{cadence}_k$ --- plafonné par famille.

**\[D\]** Fenêtre glissante continue, jamais une boucle discrète « planifier tout $\to$ exécuter un pas $\to$ tout rejouer » : à $t$, une séquence est déjà prédite (action$_1$, état$_2$, action$_2$, état$_3,\dots$, offsets relatifs, §[11](#sec:memoire){reference-type="ref" reference="sec:memoire"}.1). À $t{+}1$, action$_1$ devient l'action réellement exécutée, l'état$_2$ réel remplace sa version prédite à l'offset qui atteint $0$. Le masquage n'est pas une règle arbitraire : c'est le fait que le futur proche est encore inconnu ; il se lève tick par tick, au rythme où le réel remplace le prévu dans le tampon. Aucune replanification complète : l'orchestrateur décale sa fenêtre, les entrées aux offsets $>0$ restent en place jusqu'à maturation.

**\[P\]** Chaque prédiction sert deux fois : une fois pour la décision d'action immédiate à son émission (offset $+h^{(k)}$), une fois pour l'évaluation de sa propre justesse à sa maturation (offset $0$) --- même entrée du tampon, deux lectures à deux instants, aucune duplication de mécanisme.

Allocation dynamique de la capacité d'attention {#sec:allocation}
===============================================

Budget total $W$ (§[10.3](#sec:parallele){reference-type="ref" reference="sec:parallele"}) partagé entre $n$ fils/rôles concurrents, chacun porteur d'une urgence $u_k(t)$ (réutilisation des signaux de besoin et de surprise déjà définis, §[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}, §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}).

$$w_k(t)=\Big\lfloor W\cdot\frac{u_k(t)}{\sum_j u_j(t)}\Big\rfloor,\qquad \sum_k w_k(t)\le W$$

**\[P\]** Allocation proportionnelle sous ressource bornée, cf. ordonnancement équitable pondéré (*Weighted Fair Queueing*, Demers, Keshav & Shenker, 1989). Fondement cognitif de la réallocation par urgence : théorie de la capacité attentionnelle limitée (Kahneman, 1973).

**\[D\]** Un seul substrat de calcul (un Set Transformer, une matrice $\Omega,\mathcal{E}$), capacité réallouée par rôle plutôt que plusieurs mécanismes distincts --- hypothèse retenue par défaut, alternative (mécanismes hiérarchiques distincts) non exclue, non tranchée.

**\[D\]** La création de module (§[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}.5, §[10.2](#sec:triplet){reference-type="ref" reference="sec:triplet"}) est un fil concurrent supplémentaire de ce mécanisme, portant une urgence $u_k$ dérivée de la surprise validée (réemploi §[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}, §[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}), servi par WFQ comme tout autre rôle, sous le garde-fou câblé (§[15](#sec:action){reference-type="ref" reference="sec:action"}.3) qui reste prioritaire absolu. Partage jour/nuit : jour = minimum viable (capter le contexte, initialiser $E,G$ et $S_{\text{new}}$, quelques pas de gradient sur l'exemple réel, amorçage inchangé §[8](#sec:historique){reference-type="ref" reference="sec:historique"}.2) ; nuit = gros de l'entraînement via $S_{\text{new}}$ à $\beta$ élevé (réemploi $\bar g_i$, §[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.3).

Opérateurs de calcul natifs
===========================

Catalogue restreint aux primitives de jeu d'instructions matériel : addition, multiplication, comparaison, décalage de bits, lecture/écriture mémoire --- jamais de fonction transcendante câblée.

**\[P\]** Les fonctions transcendantes (sin, log, sqrt) s'obtiennent par itération de primitives, pas comme primitives elles-mêmes --- CORDIC (Volder, 1959) : calcul historique de sin/cos/sqrt sur matériel par addition et décalage en boucle, preuve d'existence que ces fonctions émergent de la composition, non l'inverse.

Garde-fous :

-   Domaine de validité : $\text{div}(a,b):=a/(b+\epsilon\,\text{sign}(b))$ si $|b|<\text{seuil}$ ; clamps analogues pour toute opération à singularité.

-   Compositions sûres (exposants entiers fixes, linéaires dans les paramètres --- problème convexe, régression linéaire sur features transformées) vs. risquées (exposant appris, divisions en cascade --- mêmes précautions qu'un MLP).

-   Choix DISCRET de la composition, jamais relaxé en continu (pas de DARTS/Gumbel-softmax) --- cohérent avec le traitement discret de toute décision structurelle (§3.3, §[9](#sec:rebranch){reference-type="ref" reference="sec:rebranch"}). Seuls les coefficients scalaires, une fois la forme choisie, sont optimisés par gradient.

-   Catalogue volontairement petit, universel --- jamais une fonction déjà résolvante du problème visé.

Décision d'action {#sec:action}
=================

Fusion pondérée, pas remplacement
---------------------------------

**\[D\]** La sélection d'action utilise une combinaison continue, pondérée par confiance, de la perception réelle et de la génération prédite (mode fusion, $W_{\text{fusion}}$) --- jamais une substitution binaire de l'un par l'autre. Le processus de réconciliation (§[4](#sec:surprise){reference-type="ref" reference="sec:surprise"}) tourne en continu, indépendamment de cette pondération, et alimente en retour la confiance qui la détermine.

Récompense intrinsèque
----------------------

$$r^{\text{intrinsèque}}_t=L_{\text{total},i}(t-1)-L_{\text{total},i}(t)$$
Capture identiquement une baisse d'erreur à complexité égale et une baisse de complexité à erreur égale (réemploi direct de §3.3).

Priorisation par besoin dominant
--------------------------------

**\[D\]** Besoin actif à $t$, hystérésis de marge $\delta>0$ :
$$k_t=\begin{cases}k_{t-1} & \text{si }\nexists\,k'\neq k_{t-1}:\ b_t[k']>b_t[k_{t-1}]+\delta\\[4pt]\arg\max_k b_t[k] & \text{sinon}\end{cases}$$
sur l'ensemble des besoins $b_t$ (§[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}). Sélection d'action gouvernée par le seul besoin actif :
$$a^*_t=\arg\max_a u_{k_t}(a)$$
Remplace une priorisation continue pondérée : un besoin gouverne à la fois, pas de mélange des composantes de $b_t$. Garde-fou non appris (réflexe de douleur, câblé) prioritaire sur $k_t$, évalué avant la sélection par besoin dominant.
[\[sec:gardefou\]]{#sec:gardefou label="sec:gardefou"}

**\[D\]** \[Non résolu, marqué explicitement\] Génération des actions candidates à horizon $>1$ non spécifiée dans les versions précédentes. Piste cohérente avec le document : actions candidates énumérées depuis $\mathcal{A}$ (discret, borné, §[1](#sec:cadre){reference-type="ref" reference="sec:cadre"}.1), évaluées par rollout dans les tampons multi-échelle (§[12](#sec:multiechelle){reference-type="ref" reference="sec:multiechelle"}) via la fusion (§[15](#sec:action){reference-type="ref" reference="sec:action"}.1), $u_k(a)$ = valeur prédite du besoin actif $k$ à maturation. Réutilisation stricte de mécanismes existants si retenue ; case laissée ouverte, limite reportée au tableau §16.

Synthèse
========

  **Mécanisme**                         **Fondement**                                                                                                                                                   **Garantie**                                                  **Limite**
  ------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------- ----------------------------------------------------
  Entraînement local                    BCD (Tseng, 1993)                                                                                                                                               Décroissance locale, $L$-lisse                                Pas de convergence globale
  Croissance structurelle               Monotonie par morceaux ($A_t$ fixé)                                                                                                                             Décroissance locale entre événements                          Pas de convergence globale (support variable)
  Disponibilité anticipée               Plateau (Page, 1954)                                                                                                                                            Critère opérationnel                                          Régime non stationnaire
  Verrouillage asymétrique              Non-infériorité (réemploi)                                                                                                                                      Plancher protégé                                              ---
  Rejet par gate                        Wilks (1938)                                                                                                                                                    $\chi^2$ asymptotique                                         Faible échantillon
  Compression                           Shannon, MDL (Rissanen)                                                                                                                                         $R(D)$, MDL consistant                                        Proxy heuristique
  Multi-fidélité                        Non-unicité de $R(D)$                                                                                                                                           Cohérent Shannon                                              Choix de $D$ non spécifié
  Croissance gloutonne                  MDL incrémental                                                                                                                                                 Réduction acceptée                                            Piège glouton (Lottery Ticket)
  Surprise                              Neyman-Pearson, Mahalanobis                                                                                                                                     Seuillage optimal                                             $H_1$ inconnue
  Récurrence                            SPRT (Wald)                                                                                                                                                     Optimalité (Wald-Wolfowitz)                                   $p_0,p_1$ bien estimées
  Cadence variable                      ---                                                                                                                                                             ---                                                           Choix de conception, non testé
  One-shot                              NCE, few-shot bayésien                                                                                                                                          Cohérence asymptotique                                        Puissance faible $n=1$
  Atténuation douce                     Shrinkage (James-Stein, LASSO)                                                                                                                                  Gradient non tué                                              Seuil $\lambda$ arbitraire
  Coût de branchement                   VC/Rademacher                                                                                                                                                   Majorant généralisation                                       Borne, pas valeur exacte
  Recherche A$^*$                       A$^*$, best-first pondéré                                                                                                                                       Optimal si $h$ admissible                                     $h$ appris, optimalité perdue
  Historique/rejeu                      Bandits contextuels, Dyna                                                                                                                                       Réduit biais de sélection                                     Fidélité des générateurs figés
  Augmentation générative               Extension du rejeu                                                                                                                                              Few-shot RL sur 1 succès                                      Négatif imaginé = mémoire réelle indiscernable
  Attention modulaire                   Set Transformer                                                                                                                                                 Invariance par permutation                                    ---
  Sortie (triplet)                      Pointer Network                                                                                                                                                 Vocabulaire variable géré                                     ---
  Synchronisation                       Dataflow (Dennis)                                                                                                                                               Cohérence du planning parallèle                               ---
  Transfert d'opérateur                 Analogie linéaire, structure-mapping                                                                                                                            Observée ailleurs                                             Hypothèse à tester ici
  Mémoire de travail                    Ligne à retard                                                                                                                                                  Décalage relatif automatique                                  Taille bornée
  Mémoire de certitude                  CLS (McClelland et al.)                                                                                                                                         Séparation rapide/lente                                       ---
  Allocation attention                  WFQ, Kahneman (1973)                                                                                                                                            Allocation proportionnelle                                    Substrat unique vs. multiple non tranché
  Opérateurs natifs                     ISA, CORDIC                                                                                                                                                     Composition exacte, bon marché                                Choix discret non différentiable
  Fusion action                         $W_{\text{fusion}}$ pondéré                                                                                                                                     Continu, jamais un remplacement dur                           Poids de fusion non prouvés optimaux
  Besoin dominant                       Hystérésis (argmax + marge $\delta$)                                                                                                                            Un seul besoin actif, anti-oscillation                        Marge $\delta$ non calibrée
  Fiabilité par contexte                Extension contextuelle de $\pi_i$ (§1.4)                                                                                                                        Détection indépendante du temps                               Coût de calcul par contexte
  Accumulateur de gradient              Moment (Polyak, 1964 ; Kingma & Ba, 2014)                                                                                                                       État unique jour/nuit, deux cadences                          Hyperparamètres $\beta$ non calibrés
  Activation creuse                     Codage parcimonieux (Olshausen & Field, 1996)                                                                                                                   Catalogue surcomplet, calcul borné par $W$                    Dictionnaire d'opérateurs non appris explicitement
  Pipeline réparation/recherche         Réemploi §1, §2, §5, §7, §8, §9                                                                                                                                 Fail-into-creation si simulacre implausible et SPRT franchi   Orchestration seule pour le cas plausible
  Création jumelée                      SPRT réemployé + CLS (§[10.8](#sec:credit){reference-type="ref" reference="sec:credit"}, §[10.2](#sec:triplet){reference-type="ref" reference="sec:triplet"})   Déclenchement séquentiel, certification différée              $D_\phi$ hors distribution au moment critique
  Regret de composition                 Rejeu contrefactuel réemployé (§[10.8](#sec:credit){reference-type="ref" reference="sec:credit"})                                                               Sépare faute module / faute orchestrateur                     Approximation jour via $V_\psi$, pas exacte
  Gradient de politique des pointeurs   REINFORCE (Williams, 1992)                                                                                                                                      Baseline = regret, réduit la variance                         Pas de garantie de convergence en non-linéaire
  Ancrage de composition                Test de non-infériorité réemployé (§9)                                                                                                                          Résidu toujours mesurable (réel ou module certifié)           Choix de niveau non calibré empiriquement
  A$^*$ ancrée                          Réemploi §7.1--7.2, §15.1                                                                                                                                       $g(n),h(n)$ évalués sur un point vérifiable                   Coût de la cascade de génération par nœud
  Fenêtre glissante continue            Extension du tampon relatif (§12)                                                                                                                               Pas de replanification complète, chaque tick                  Aucune nouveauté, clarification seule
  Accumulateur orchestrateur            Moment (Polyak ; Kingma-Ba), réemployé                                                                                                                          Distinct du gradient module, même recette                     $\beta$ non calibrés spécifiquement

Conclusion
==========

Corps du document strictement mathématique : chaque mécanisme, son fondement, sa garantie, sa limite. Justifications, motivations et exemples : Annexe A. Dimensions proposées : Annexe B.

La v6 ferme deux organes manquants de la v5 : le crédit des décisions discrètes de composition (§[10.8](#sec:credit){reference-type="ref" reference="sec:credit"}, §[10.2](#sec:triplet){reference-type="ref" reference="sec:triplet"}, REINFORCE et regret de composition) et la porte de la nouveauté radicale (SPRT de création, localisation du point de branchement, création jumelée module + simulateur, statut provisoire jusqu'à confirmation réelle).

Justifications de design
========================

Style télégraphique. Une idée, une justification, par entrée. Renvoi à la section du corps principal entre parenthèses.

**Besoins réduits à faim/ennui (§1.1).** Fatigue inutile ici (nuit imposée par le simulateur). Peur et créativité hors périmètre initial (pas de danger, pas d'objectif de production dans le monde actuel) --- réintroductibles plus tard sans changer le mécanisme, même vecteur, même sélection. Incertitude fusionnée dans le sens d'ennui plutôt que gardée séparée --- un signal de moins à faire interagir.

**Priorité par argmax, pas somme (§1.1, §15.3).** Manger et explorer en même temps n'a pas de sens comportemental --- un seul besoin doit gouverner l'action à chaque instant. Une somme pondérée aurait mélangé les deux en continu. L'hystérésis évite qu'un quasi-ex-æquo fasse osciller la décision à chaque pas.

**Contexte vs pointable (§1.2).** L'orchestrateur doit savoir ce qui l'entoure (condensateurs, besoins) sans pouvoir pointer dessus comme source/cible d'un triplet --- sinon le pointeur confondrait un signal de conditionnement avec un flux de contenu.

**Sortie légèrement plus large que l'entrée (§1.2).** Sans un canal réinjecté, l'orchestrateur n'aurait aucune trace de ce qu'un module vient de produire au-delà de sa consommation immédiate --- ce canal réinjecté est ce qui permet de \"savoir à quoi on pense\" sans stocker toute la pensée.

**Fiabilité par contexte, pas seulement globale (§1.4).** Un module peut réussir en moyenne et échouer systématiquement sur une configuration précise --- le condensateur global masquerait ce cas. $\pi_i(x)$ le rend visible, contexte par contexte.

**Un seul discriminateur, pas un par module (§1.3, §5).** Dupliquer $D_\phi$ à chaque module ajouterait un mécanisme entier sans gain : juger si un contenu est plausible ne dépend pas du module qui l'a produit.

**Module visuel de départ donné, pas découvert (§1.3).** Rien ne garantit qu'un orchestrateur naïf invente seul un bon compresseur d'image dès le départ --- cette brique est donnée, dimensionnée et entraînée par reconstruction masquée, comme des yeux déjà câblés plutôt que des yeux à assembler seuls.

**Vitesse→image, premier exemple filé (§1.3).** Utile de fixer un cas concret dès le corps du texte : la prédiction de vitesse alimente directement le générateur d'image suivante --- sans ce chaînage explicite dès le départ, rien ne force les modules à s'articuler entre eux.

**Accumulateur de gradient partagé jour/nuit (§1.3).** Recalculer un état d'apprentissage séparé pour la nuit dupliquerait la mémoire des poids. Un seul accumulateur, alimenté à cadence différente selon l'heure, garde une seule source de vérité par module.

**Échantillon varié plutôt que fenêtre glissante (§1.4).** Un module peut sembler stable sur des mesures consécutives simplement parce que rien ne change dans son contexte immédiat --- la variété de l'échantillon teste réellement sa robustesse.

**Dégradation → signal, pas rejet silencieux (§1.4).** Un exemple qui dégrade un module ne doit pas juste être ignoré : il doit apprendre au prédicteur de fiabilité à le reconnaître, sinon le même échec se reproduit indéfiniment sans que l'orchestrateur en soit informé.

**Local vs global, laissé ouvert au code (§2).** La théorie retient le local strict par choix de rigueur, mais rien n'empêche de tester empiriquement le gradient global une fois le code en main --- les deux mécanismes ne coûtent pas cher à comparer.

**Surprise = contexte, pas horloge (§4).** Chercher \"ce qui vient de changer\" suppose de regarder le temps ; chercher \"où la fiabilité est basse\" fonctionne aussi bien pour une situation rare rencontrée pour la première fois après un an que pour un changement immédiat.

**Implausibilité confirmée = création, pas abandon (§4.5).** Un phénomène radicalement inédit est par définition hors de portée générative des modules actuels --- le jeter reviendrait à jeter précisément les cas où *\[NEW\]* devrait se déclencher. L'implausibilité ne devient un signal de création qu'une fois confirmée par un SPRT sur plusieurs échecs distincts, pas sur un seul essai.

**Localisation par premier $\pi$ effondré à antécédents sains (§4.6).** Accuser le bon coupable dans une chaîne exige de tester chaque maillon sur le même $x$, pas sur sa moyenne --- sinon la dégradation en aval fait porter le blâme au mauvais module.

**Simulateur = mémoire épisodique générative (§10.2).** Stocker l'épisode brut coûte cher et ne généralise pas ; stocker une machine capable de le refabriquer permet des variations à chaque rejeu plutôt qu'une copie figée --- c'est ce qui rend le rejeu nocturne utile à partir d'un seul exemple réel.

**Statut provisoire tant que non confirmé par le réel (§1.4, §9).** Certifier un module sur la seule base de son propre simulateur reviendrait à valider une hypothèse par elle-même --- seule une occurrence réelle ultérieure peut lever le doute.

**Création = rôle WFQ explicite (§13).** Un contexte surprenant est par nature potentiellement critique ; sans budget déclaré, l'entraînement d'un module neuf entrerait en concurrence non arbitrée avec les autres fils au pire moment.

**Bit de provenance et purge (§8.3).** Sans marquage réel/imaginé, une confabulation jamais confirmée s'accumule indéfiniment dans la mémoire du gating. La désuétude, déjà prévue pour tout module inutile, suffit à l'effacer --- pas besoin d'un mécanisme d'oubli séparé.

**Regret de composition comme baseline (§10.8).** Le rejeu contrefactuel nocturne calcule déjà, pour chaque candidat, ce qu'aurait donné l'alternative --- autant s'en servir comme référence pour départager la faute du module de la faute du choix.

**REINFORCE plutôt qu'une relaxation continue (§10.2).** Le §14 interdit déjà de relâcher les choix structurels en continu ; un gradient de politique respecte cette discrétion tout en fournissant un signal, même bruité, à travers la chaîne de décisions.

**Masque de compatibilité de type (§10.2).** Une incompatibilité connue à l'avance (type source vs opérateur) ne doit pas être redécouverte par un échec coûteux --- l'exclure du softmax est immédiat et gratuit.

**Monotonie par morceaux, pas globale (§2.2).** Prétendre à une convergence globale sur un ensemble de modules dont le cardinal change serait une affirmation non fondée --- la garantie ne vaut qu'entre deux événements structurels, à catalogue fixé.

**Contrôle de multiplicité des tests (§9).** Des dizaines de tests de non-infériorité par jour sans correction laissent s'accumuler des faux positifs qui érodent la confiance dans les planchers \"certifiés\".

**Le verrouillage protège du passé, pas de l'avenir (§9).** Un plancher mesuré sur une distribution devenue obsolète après un drift durable ne devrait pas bloquer une adaptation nécessaire --- d'où le re-calage conditionné au SPRT de nouveauté.

**Génération d'actions, case ouverte mais visible (§15.3).** Mieux vaut marquer honnêtement un trou que le combler par une affirmation non vérifiée --- la piste proposée réutilise strictement les mécanismes déjà posés.

**Catalogue surcomplet, activation creuse (§10.7).** Avoir beaucoup plus d'outils disponibles que d'outils utilisables à un instant donné n'est pas un défaut à corriger --- c'est ce qui permet la spécialisation sans exploser le calcul par pas.

**Modules endormis, pas un mécanisme de rêve séparé (§10.7).** Garder un contexte gênant en mémoire jusqu'à la nuit réutilise exactement l'opérateur idle déjà défini pour la synchronisation --- un tampon dédié aurait été redondant.

**Éléments de $T_t$ toujours issus d'un module (§10.1).** Pointer vers une valeur brute non compressée quand un module sait déjà la résumer serait un retour en arrière --- la compression disponible doit toujours être préférée au brut quand elle existe.

**Profondeur de prévision = conséquence, pas capacité (§10.1).** L'orchestrateur ne devient pas meilleur prédicteur en soi --- il profite seulement de modules qui compressent mieux. Séparer les deux évite de chercher à améliorer le mauvais composant quand une prédiction lointaine échoue.

**L'orchestrateur compose, ne calcule pas (§10.2).** Toute valeur produite doit provenir d'un module --- sinon la distinction module/orchestrateur perd son sens et le gradient local (§2) n'a plus de destinataire clair.

**Ancrage obligatoire de toute composition (§7.4).** Une composition jamais comparée au réel ou à un module certifié ne reçoit aucun signal d'apprentissage --- c'était le trou de la version précédente. Sans ce point de sortie vérifiable, tout le reste (recherche A\*, gating, accumulateurs) tourne à vide.

**Arrêt anticipé de la cascade via non-infériorité (§7.4).** Redescendre systématiquement jusqu'au pixel serait un gaspillage si un module intermédiaire déjà certifié suffit --- même test déjà utilisé pour le verrouillage, pas de mécanisme nouveau à inventer.

**A$^*$ ancrée (§7.5).** Sans point de comparaison vérifiable à chaque nœud, $g(n)$ et $h(n)$ seraient évalués sur du vent --- l'ancrage rend la recherche elle-même évaluable, pas seulement la décision finale.

**Accumulateur de l'orchestrateur, distinct de celui du module (§10.8).** Un module peut bien fonctionner alors que le choix de le brancher ici était mauvais, et inversement. Séparer les deux résidus évite d'accuser le mauvais coupable quand une prédiction échoue.

**Verrouillage = plancher, pas plafond (§1.4).** Sans ça, absence de gradient global fige tout module dès qu'il atteint un score moyen. Un module peut continuer de progresser si ses antécédents/successeurs évoluent --- seule la régression est interdite.

**Rejet précédé d'un essai de gating (§2.3).** Un module rare mais localement décisif serait sinon rejeté à tort sur sa moyenne globale. On tente d'abord de lui trouver un contexte de niche avant d'abandonner.

**Multi-fidélité (§3.2).** Une image compressée à l'extrême ne redonne jamais un mot, et un mot ne redonne jamais une image fidèle. Plusieurs niveaux de compression coexistent, chacun pour son usage --- pas un seul niveau optimal universel.

**Mise en garde croissance gloutonne (§3.3).** Une capacité temporairement plus grande peut être le seul chemin vers une compression finale meilleure. Croissance pas-à-pas risque de rater ce chemin. Non tranché : décision à prendre au déploiement.

**Atténuation douce, pas suppression (§5).** Mettre une dimension à zéro tue son gradient --- plus jamais récupérable si l'évidence future la concerne. Un poids qui décroît sans jamais atteindre zéro reste réversible.

**Rejeu contrefactuel (§8.1).** Se contenter du module réellement choisi laisse un biais de sélection --- on ne sait jamais ce qu'aurait donné l'alternative. Rejouer tous les candidats à poids figés, à coût d'inférence, corrige ce biais sans coût réel supplémentaire.

**Amorçage à la création (§8.2).** Un module neuf a un embedding non façonné --- sans amorçage, l'attention n'a aucune raison de le retrouver avant une redécouverte par hasard. On force un premier exemple positif dès la création.

**Augmentation générative sur un seul succès (§8.3).** Un événement réel rare (réussite de conduite dans le vent) ne suffit pas statistiquement. On génère, par les générateurs déjà figés, ce qu'aurait donné l'alternative non choisie, pour construire un exemple négatif immédiat plutôt que d'attendre qu'il se reproduise réellement. Conséquence acceptée : un événement négatif purement imaginé s'enregistre dans le même format qu'un événement réel --- même mécanisme qui produit la mémoire, produit aussi une croyance jamais vécue.

**Triplet (source, opérateur, cible) (§10.2).** Le catalogue change de taille chaque jour ; un vocabulaire de sortie fixe est impossible. Un pointeur sur les éléments courants résout ça directement, sans réentraîner une couche de sortie à chaque nouveau module.

**Opérateur idle/NOP à deux usages (§10.3).** Le même objet --- ne rien faire --- sert à la fois de bulle de synchronisation (attendre qu'une dépendance soit prête) et de maintien en mémoire de travail (garder un contenu sans le modifier). Un seul objet, deux usages, pas deux mécanismes à maintenir.

**Trace auto-référentielle (§10.6).** La séquence de décisions passées est un flux comme un autre --- rien n'empêche l'attention de s'appliquer à sa propre trace. Aucune affirmation sur la conscience n'est faite ici : seul le mécanisme (un flux de plus dans le catalogue) est formalisé.

**Mémoire de travail en décalage relatif, pas en horaire absolu (§11.1).** Un index par horaire absolu grossit sans borne conceptuelle et complique le glissement d'une prédiction à l'autre. Un tampon de taille fixe, indexé par position relative à l'instant présent, fait glisser les prédictions sans logique ajoutée --- le décalage est une propriété du tampon, pas un calcul supplémentaire.

**Deux mémoires séparées (§11).** Une croyance de longue date (\"travailler dur donne un bon métier\") n'est pas stockée comme les poids d'un réseau ni comme une image en mémoire tampon --- elle a son propre statut : un chemin de modules, mesuré lentement, contredit lentement. Séparer mémoire de travail (rapide, bornée) et mémoire de certitude (poids, lente) évite de confondre les deux régimes.

**Prédiction multi-échelle (§12).** Une prédiction \"dans une minute\" et une prédiction \"demain\" ne doivent pas partager la même cadence de vérification --- sinon soit le court terme est sous-vérifié, soit le long terme sature la mémoire. Plusieurs familles, chacune sa cadence, le même mécanisme de fond.

**Fenêtre glissante continue, pas de replanification totale (§12).** Tout rejouer à chaque pas jetterait un travail de prédiction déjà valide sur les offsets non encore mûris. Le tampon à décalage relatif suffit déjà à ne garder que ce qui reste à confirmer --- le masquage se lève de lui-même, pas besoin d'une règle de reprise séparée.

**Chaque prédiction utilisée deux fois (§12).** La même entrée du tampon sert d'abord à décider l'action, puis plus tard à s'auto-évaluer une fois mûrie --- deux lectures, un seul stockage, pas de duplication.

**Allocation dynamique de l'attention (§13).** Faire plusieurs choses à la fois (écouter, conduire, parler) puis se concentrer soudain sur un danger n'exige pas plusieurs cerveaux séparés --- un budget de calcul partagé, réalloué par urgence, explique le phénomène avec un seul mécanisme.

**Opérateurs natifs = ISA matériel, pas fonctions transcendantes (§14).** Personne n'a besoin de calculer un logarithme pour vivre. Les fonctions utiles émergent de la composition d'opérations élémentaires (addition, comparaison, boucle) --- les donner toutes faites reviendrait à injecter une compétence non universelle.

**Fusion pondérée plutôt que remplacement perception/imagination (§15.1).** Agir sur l'imagination à la place du réel, sans garde-fou, n'a jamais été validé comme principe général. Une combinaison continue, pondérée par la confiance mesurée, et une vérification permanente indépendante de cette pondération, est le choix retenu.

Dimensions proposées
====================

  **Objet / variable**                                     **Valeur proposée**                                           **Justification**                                                                                                                                     **Statut**
  -------------------------------------------------------- ------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------------------- ------------------
  $\dim_{\text{emb}}$ (embedding de contenu)               32--64                                                        Goulet d'information du pointeur : le softmax de sélection ne peut pas discriminer plus de configurations que l'embedding n'en encode (relevé de 8)   recalé
  $\dim_{\text{op}}$ (embedding d'opérateur)               8--16                                                         Même ordre de grandeur que $\dim_{\text{emb}}$, à recaler empiriquement                                                                               nouveau
  $d_{\text{model}}$ (dimension interne Set Transformer)   64                                                            Volontairement sur-dimensionné dès le départ (cf. cold start)                                                                                         nouveau
  Nombre de têtes d'attention                              4--8                                                          Choix standard pour un modèle de cette taille                                                                                                         nouveau
  $W$ (largeur parallèle, macro-pas)                       16--32                                                        \"Des dizaines\" de triplets simultanés, borné par la capacité de calcul réelle                                                                       nouveau
  $K$ (portée mémoire de travail, offsets $\pm K$)         8--16                                                         Extension du n\_frames=3 actuel une fois les latents compressés disponibles                                                                           étend l'existant
  Familles d'horizon $\mathcal{H}$                         court $\{1,2,3\}$, moyen $\{10,30,100\}$, long événementiel   Court terme = existant (periode\_eval\_prevision) ; moyen/long = nouveau                                                                              mixte
  Nombre de types $\tau$                                   5                                                             spatial-$x$, spatial-$y$, temporel, intensité, catégoriel --- minimal, révisable                                                                      nouveau
  $\epsilon_s$ (seuil pente plateau)                       à calibrer, ex. 1e-4                                          Même ordre que seuil\_variation\_apprentissage existant                                                                                               nouveau
  $\epsilon_\sigma$ (stabilité du bruit)                   à calibrer                                                    Pas de précédent direct dans le code actuel                                                                                                           nouveau
  Seuil non-infériorité $\Delta$                           à calibrer, ex. 0.05                                          Cohérent avec seuil\_integration existant (0.1) en ordre de grandeur                                                                                  nouveau
  Seuil critique $\chi^2$ (Wilks)                          selon $df$ et $\alpha=0.05$                                   Valeur tabulée standard, pas un paramètre à apprendre                                                                                                 nouveau
  $(\alpha,\beta)$ SPRT                                    0.05, 0.10                                                    Valeurs conventionnelles en tests séquentiels                                                                                                         nouveau
  Catalogue d'opérateurs natifs                            $\sim$10--15 primitives                                       addition, multiplication, comparaison, décalage, quelques constructions de contrôle                                                                   nouveau
  $d_i^{\text{réinj}}$ (canal réinjecté)                   2--4                                                          Petite fraction de $d_i'$, ne doit pas dominer le budget de sortie                                                                                    nouveau
  CNN visuel par défaut                                    4 blocs conv, latent 32--64                                   Ordre de grandeur standard pour un champ visuel réduit (toy world)                                                                                    nouveau
  $\beta$ (moment accumulateur $\bar g_i$)                 0.9 (jour), 0.99 (nuit)                                       Valeurs conventionnelles type Adam (Kingma & Ba, 2014)                                                                                                nouveau
  $\delta$ (hystérésis besoin dominant)                    à calibrer, ex. 0.05                                          Même ordre que `seuil_integration` existant                                                                                                           nouveau
  $\beta$ (accumulateur $\bar g_{\text{orch}}$)            mêmes valeurs que $\bar g_i$                                  Même recette, appliquée au niveau orchestrateur                                                                                                       nouveau
  $(\alpha,\beta)$ SPRT de création                        0.05, 0.10                                                    Mêmes valeurs conventionnelles que le SPRT de surprise (§4.3)                                                                                         nouveau
  Plafond ratio imaginé/réel (gating)                      3:1                                                           À calibrer ; évite la saturation du lot par la confabulation                                                                                          nouveau
  Dimensionnement $S_{\text{new}}$                         même ordre que le module jumeau                               Cohérence de taille avec $M_{\text{new}}=(E,G)$                                                                                                       nouveau

99
Assran, M. et al. (2023). *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR.
Bartlett, P. & Mendelson, S. (2002). *Rademacher and Gaussian Complexities*. JMLR.
Benjamini, Y. & Hochberg, Y. (1995). *Controlling the False Discovery Rate*. JRSS-B.
Blackwelder, W.C. (1982). *Proving the null hypothesis in clinical trials*. Controlled Clinical Trials.
Bottou, L., Curtis, F. & Nocedal, J. (2018). *Optimization Methods for Large-Scale Machine Learning*. SIAM Review.
Burges, C. et al. (2005). *Learning to Rank using Gradient Descent*. ICML.
Demers, A., Keshav, S. & Shenker, S. (1989). *Analysis and Simulation of a Fair Queueing Algorithm*. SIGCOMM.
Dennis, J.B. (1974). *First Version of a Data Flow Procedure Language*. MIT.
Dudík, M., Langford, J. & Li, L. (2011). *Doubly Robust Policy Evaluation and Learning*. ICML.
Frankle, J. & Carbin, M. (2018). *The Lottery Ticket Hypothesis*. ICLR.
Gentner, D. (1983). *Structure-Mapping: A Theoretical Framework for Analogy*. Cognitive Science.
Gutmann, M. & Hyvärinen, A. (2010). *Noise-Contrastive Estimation*. JMLR/AISTATS.
Hart, P., Nilsson, N. & Raphael, B. (1968). *A Formal Basis for the Heuristic Determination of Minimum Cost Paths*. IEEE Trans. SSC.
Hinton, G., Vinyals, O. & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. NeurIPS Workshop.
Kahneman, D. (1973). *Attention and Effort*. Prentice-Hall.
James, W. & Stein, C. (1961). *Estimation with Quadratic Loss*. Berkeley Symposium.
Kingma, D.P. & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. ICLR.
Lake, B., Salakhutdinov, R. & Tenenbaum, J. (2015). *Human-level concept learning through probabilistic program induction*. Science.
LeCun, Y. (2022). *A Path Towards Autonomous Machine Intelligence*. OpenReview.
Lee, J. et al. (2019). *Set Transformer*. ICML.
Li, L., Chu, W., Langford, J. & Schapire, R. (2010). *A Contextual-Bandit Approach to Personalized News Article Recommendation*. WWW.
McClelland, J., McNaughton, B. & O'Reilly, R. (1995). *Why there are complementary learning systems in the hippocampus and neocortex*. Psychological Review.
Mikolov, T. et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.
Neyman, J. & Pearson, E. (1933). *On the Problem of the Most Efficient Tests of Statistical Hypotheses*. Phil. Trans. Royal Society.
Olshausen, B. & Field, D. (1996). *Emergence of simple-cell receptive field properties by learning a sparse code for natural images*. Nature.
Oppenheim, A. & Schafer, R. *Discrete-Time Signal Processing*. Prentice-Hall.
Page, E.S. (1954). *Continuous Inspection Schemes*. Biometrika.
Pohl, I. (1970). *Heuristic search viewed as path finding in a graph*. Artificial Intelligence.
Polyak, B.T. (1964). *Some methods of speeding up the convergence of iteration methods*. USSR Computational Mathematics and Mathematical Physics.
Rissanen, J. (1978). *Modeling by shortest data description*. Automatica.
Robbins, H. & Monro, S. (1951). *A Stochastic Approximation Method*. Annals of Mathematical Statistics.
Shannon, C. (1948). *A Mathematical Theory of Communication*. Bell System Technical Journal.
Shannon, C. (1959). *Coding Theorems for a Discrete Source With a Fidelity Criterion*.
Sutton, R. (1988). *Learning to Predict by the Methods of Temporal Differences*. Machine Learning.
Sutton, R. (1990). *Integrated Architectures for Learning, Planning, and Reacting*. ICML.
Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*, 2e éd., MIT Press.
Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*. JRSS-B.
Tsitsiklis, J. (1994). *Asynchronous Stochastic Approximation and Q-Learning*. Machine Learning.
Tseng, P. (1993). *Convergence of a Block Coordinate Descent Method*. J. Optimization Theory and Applications.
Vapnik, V. (1998). *Statistical Learning Theory*. Wiley.
Vinyals, O., Fortunato, M. & Jaitly, N. (2015). *Pointer Networks*. NeurIPS.
Volder, J. (1959). *The CORDIC Trigonometric Computing Technique*. IRE Trans. Electronic Computers.
Williams, R.J. (1992). *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*. Machine Learning.
Wald, A. (1945). *Sequential Tests of Statistical Hypotheses*. Annals of Mathematical Statistics.
Wald, A. & Wolfowitz, J. (1948). *Optimum Character of the Sequential Probability Ratio Test*.
Wilks, S.S. (1938). *The Large-Sample Distribution of the Likelihood Ratio for Testing Composite Hypotheses*. Annals of Mathematical Statistics.
