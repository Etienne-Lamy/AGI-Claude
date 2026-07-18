"""Hyperparamètres SCL. Aucune catégorie sémantique ici — uniquement des scalaires.

Fichier vivant : chaque phase de la réécriture v6 (voir plan de réécriture)
ajoute exactement les constantes dont son propre fichier a besoin, plutôt que
de précharger tout l'Annexe B d'un coup. Phase 0 : nettoyage des clés liées
aux mécanismes abandonnés (pilote de chantiers, primitive slots, instrument
des barreaux, orchestrateur à mélange continu des besoins) — tout le reste
est conservé tel quel en attendant la phase qui le retouche explicitement.
"""

CONFIG = dict(
    # --- dimensions (module.py, Phase 2) ---
    dim_emb=8,                     # espace latent commun des embeddings de modules
    dim_contexte=16,               # taille du contexte enrichi (boucle.py, Phase 11)
    n_hidden_init=8,
    croissance_pas=4,              # unités cachées ajoutées par tentative de croissance
    croissance_max=128,

    # --- condensateurs et verrouillage asymétrique (module.py, §1.4, Phase 2) ---
    seuil_succes=0.05,
    delta_succes=0.02,
    delta_echec=0.05,
    seuil_verrou=0.9,
    seuil_atrophie=0.05,
    seuil_effondrement=0.15,

    # --- learning rates locaux (module.py, Phase 2) ---
    lr_normal=1e-2,
    lr_lent=1e-3,
    lr_recherche_latent=0.2,       # aligner_action, §1.3
    n_iterations_alignement=25,
    n_iterations_latent_predictif=5,   # chercher_latent_predictif, §1.3

    # --- saturation (module.py, Phase 2) ---
    seuil_grad_bas=1e-4,
    seuil_erreur_mauvais=0.2,
    fenetre_friction=20,

    # --- cooldown de création (memoires.RegistreRupture, §1.4, Phase 1) ---
    delai_refroidissement=200,

    # --- croissance gouvernée (graphe.py, §2.2, Phase 4) ---
    seuil_gain_croissance=1e-3,

    # --- honnêteté de l'apprentissage (module.py / graphe.py, Phases 2 et 4) ---
    # en dessous de cette variation d'entrée, il n'y a RIEN à prédire :
    # ni entraînement, ni condensateur (empêche le verrouillage de fausses
    # compétences sur un monde statique trivialement prévisible)
    seuil_variation_apprentissage=1e-4,
    # erreur générative RELATIVE à la baseline de persistance ('rien ne
    # change') — ne pas prédire mieux que la persistance = erreur >= 1
    cap_erreur_relative=2.0,
    seuil_succes_gen_relatif=0.5,  # succès = prédire 2x mieux que 'rien ne change'
    # anti-collapse : le condensateur reco ne monte que si les latents sont
    # vivants (des latents effondrés vers une constante 'prédisent' trivialement)
    seuil_dispersion_latente=1e-3,
    max_echecs_decoupe=2,          # après 2 découpes stériles : vrai candidat
    # maturité structurelle : fragmenter exige un module qui a VÉCU (un
    # effondrement suppose une certitude passée) ; idem pour l'atrophie nocturne
    maturite_structurelle=500,

    # --- module visuel (module_visuel.py, CNN, Phase 2) ---
    conv_canaux=12,

    # --- canal réinjecté et accumulateur de gradient persistant (module.py, §1.3, Phase 2) ---
    dim_reinjection=3,
    beta_jour=0.9,                 # ḡ_i ← β ḡ_i + (1-β)∇ , cadence rapide (jour)
    beta_nuit=0.99,                # même accumulateur, lissage renforcé (nuit)

    # --- discriminateur partagé D_φ (discriminateur.py, §5, Phase 3) ---
    dim_discriminateur=16,         # espace de comparaison commun (projection si hétérogène)
    n_hidden_discriminateur=16,
    lr_discriminateur=1e-2,
    lambda_attenuation=0.5,        # atténuation douce w_j = exp(-λ r_j), jamais zéro
    seuil_plausibilite=0.5,        # D_φ(x) au-delà : simulacre jugé plausible → composition tentée avant création (§4.5 étape 3)

    # --- localisation du point de branchement, non-infériorité, rejet gouverné
    # (graphe.py, §4.6, §9, §2.3, Phase 4) ---
    seuil_pi_bas=0.5,              # π(x) en dessous : fiabilité contextuelle effondrée
    delta_non_inferiorite=0.05,    # marge de dégradation tolérée (Blackwelder, 1982)
    alpha_non_inferiorite=0.05,    # confiance 1-α des tests (non-infériorité et Wilks)

    # --- disponibilité anticipée (disponibilite.py, §1.4, Phase 4) ---
    epsilon_s=1e-4,                 # seuil de pente du plateau de progrès
    epsilon_sigma=0.05,             # seuil de variance du bruit résiduel
    taille_minimale_disponibilite=5,

    # --- curiosité / motivation intrinsèque (curiosite.py, §15.2) ---
    fenetre_incertitude=20,         # fenêtre d'erreur récente pour incertitude/progrès
    incertitude_initiale=1.0,       # incertitude d'un module jamais évalué (max ⇒ attractif)
    min_vecu_maitrise=40,           # nb min d'évaluations avant de déclarer un module maîtrisé
    seuil_incertitude_maitrise=0.02,# incertitude sous laquelle un module est "maîtrisé"
    seuil_progres_maitrise=0.005,   # progrès sous lequel on considère le plateau atteint

    # --- dynamique du corps (dynamique.py) : prédicteurs action-conditionnés ---
    n_latent_dynamique=4,           # latent d'un prédicteur (v → v_suivant)
    attrait_action_inexploree=0.1,  # incertitude d'une accélération jamais tentée (attrait de découverte)
    sigma_prior_dynamique=0.5,      # confiance du prior "rien ne change" (petit ⇒ un vrai Δv surprend)
    periode_pouls=3,                # cadence (en pas) de l'instantané "pouls" pour le dashboard

    # --- statistiques : SPRT (surprise / création / drift), FDR, cadence
    # (statistiques.py, §4, M1, M10, Phase 5) ---
    alpha_sprt_surprise=0.05, beta_sprt_surprise=0.10,
    alpha_sprt_creation=0.05, beta_sprt_creation=0.10,
    alpha_sprt_drift=0.05, beta_sprt_drift=0.10,
    decalage_sprt_surprise=0.5,     # translation de H1 (en unités de d), résidu de surprise
    decalage_sprt_drift=2.0,        # translation de H1 (en écarts-types), drift
    cadences_sprt={"sensorimoteur": 1, "defaut": 1},  # une seule famille pour le POC (Phase 7)

    # --- simulateur S_new, création jumelée (simulateur.py, §10.2, §8.3, Phase 6) ---
    n_hidden_simulateur=16,
    lr_simulateur=1e-2,
    n_contrefactuels=4,             # variantes générées par generer_contrefactuel
    echelle_bruit_contrefactuel=0.3,
    seuil_hors_distribution=0.3,    # D_φ en dessous : verdict étiqueté hypothèse, pas pilier
    clip_grad_simulateur=2.0,       # norme max du gradient brut avant incorporation à ḡ
    clip_grad_pointeurs=2.0,        # idem pour l'orchestrateur (REINFORCE), évite la divergence de u
                                     # (la perte NLL hétéroscédastique peut exploser
                                     # localement quand σ devient petit — 1/σ² amplifie
                                     # le gradient ; sans ce clip, un seul pas peut
                                     # faire diverger μ et σ simultanément)

    # --- mémoire de travail (memoire_travail.py, §11, §12, Phase 7) ---
    W=16,                            # tête rapide (largeur parallèle, macro-pas)
    K=8,                             # palier lent (portée mémoire, offsets ±K)

    # --- recherche V_ψ, A* ancrée (recherche.py, §7, Phase 8) ---
    n_hidden_v_psi=16,
    lr_v_psi=1e-2,
    gamma_v_psi=0.9,
    profondeur_max_recherche=10,

    # --- attention : Set Transformer + Pointer Network (attention.py, §10, Phase 9) ---
    d_model=64,                     # dimension interne du Set Transformer
    n_tetes_attention=4,
    dim_op=8,                       # embedding d'opérateur
    lr_pointeurs=1e-2,
    seuil_incertitude_fil=5.0,
    profondeur_max_fil=5,
    # catalogue d'opérateurs MINIMAL pour le POC (operateurs_natifs.py est
    # différé, §25 étape 13) : types_sortie=None -> compatible avec tout type
    catalogue_operateurs={
        "id": {"types_sortie": None},
        "percevoir": {"types_sortie": {"latent"}},
        "predire": {"types_sortie": {"latent"}},
        "agir": {"types_sortie": {"action"}},
    },

    # --- allocation dynamique de l'attention (allocation_attention.py, §13, Phase 10) ---
    urgence_plancher=0.01,          # aucun fil ne tombe jamais exactement à 0
    part_creation_jour_min=1,       # création = minimum viable de jour (M5)

    # --- assemblage final (boucle.py, inne.py, Phase 11) ---
    periode_verification_structurelle=20,   # tous les N pas : localiser_point_branchement
    n_reve_coordonne=20,             # nb d'entrées récentes du câblage rejouées la nuit
    taille_fenetre_drift=20,         # taille des fenêtres récente/ancienne pour sprt_drift

    # --- monde (monde.py, inchangé) ---
    taille_perception=10,
    n_frames=3,
    v_max=2,
    densite_sucre=0.03,
    densite_baton=0.02,
    graine_monde=42,
    taille_chunk=16,

    # --- besoins : b_t = (faim, ennui), §1.1, Phase 1 ---
    faim_par_step=0.002,
    faim_par_vitesse=0.001,        # aller vite creuse la faim (régulation corporelle)
    recompense_sucre=0.4,          # baisse de faim en mangeant
    ennui_par_step=0.001,          # f croissante ; plafond dur 0.5 appliqué dans le code
    delta_hysteresis_besoin=0.05,  # δ du Schmitt-trigger argmax du besoin dominant, §15.3

    # --- douleur : signal câblé (pas un besoin de b_t), garde-fou §15.3 ---
    douleur_baton=0.5,
    decroissance_douleur=0.01,     # la douleur persiste (signal d'évitement)
    seuil_reflexe_douleur=0.4,     # un seul choc déclenche le réflexe câblé
    penalite_baton_navigation=5.0, # pénalité d'un pas prédit franchissant un bâton (évitement, §15.3)

    # --- modèle de prévision du corps (prevision.py, §1.3, §12) ---
    n_hidden_prevision=16,
    lr_prevision=5e-3,
    fenetre_fiabilite_prevision=50,   # fenêtre d'erreur récente pour π
    n_maj_min_prevision=100,          # apprentissages min avant de faire confiance au modèle
    echelle_fiabilite_prevision=0.1,  # échelle de conversion erreur→π (fiabilite())
    seuil_fiabilite_appris=0.8,       # π au-delà : le modèle appris pilote la navigation (sinon instinct)

    # --- disponibilité anticipée : échantillon varié 𝒲_i(t), §1.4, Phase 1 ---
    taille_echantillon_disponibilite=20,
    seuil_diversite_disponibilite=0.5,

    # --- provenance réel/imaginé, §8.3 M6, Phase 1 ---
    plafond_ratio_imagine_reel=3.0,

    # --- observation / instrumentation générique ---
    periode_log_pensee=5,          # fenêtre d'introspection tous les N steps
    delai_step=0.0,                # secondes de pause par step (observation ralentie)

    # --- divers, à ré-auditer au fil des phases qui les consomment ---
    taille_max_historique_erreurs=500,
    fenetre_erreur_globale=100,
    seuil_activation_entrainement=0.1,
)
