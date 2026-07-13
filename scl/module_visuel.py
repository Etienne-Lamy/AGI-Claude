"""Module visuel SCL — module sensoriel par défaut, donné a priori (pas
découvert par l'orchestrateur, §1.3). Interface identique à `Module` (mêmes
entraînements, condensateurs, verrous, recherche de latent prédictif) ;
seule la paramétrisation change : un encodeur/décodeur convolutif dimensionné
pour la résolution du champ visuel, plutôt que le MLP générique.

Remplace `module_slots.py`/`module_conv.py` (3 générations documentées comme
échecs dans README v2.md : bruit de permutation, échelle de coordonnées non
normalisée, décodeur entraîné sur une sortie d'encodeur bruitée plutôt que
sur la cible propre, ancrage temporel instable au démarrage). Root cause
retenue : reconstruire la vision comme un autoencodeur convolutif standard,
validé isolément (frames synthétiques, sans monde réel) avant intégration —
c'est exactement ce que `tests/test_module_visuel.py` fait.

Auto-supervision : reconstruction masquée façon JEPA (LeCun 2022 ; Assran
et al. 2023) — masque aléatoire sur le champ visuel, cible = reconstruction
COMPLÈTE et PROPRE (jamais une version bruitée produite par l'encodeur —
c'est précisément le bug documenté du décodeur de `module_slots.py`).
Encodeur et décodeur de CE module sont entraînés conjointement (un seul
passage arrière à travers les deux) : ceci reste strictement local (aucun
gradient ne sort du module) — §0 porte sur les frontières ENTRE modules, pas
sur les sous-réseaux internes d'un même module (§2.1, Φ(θ)=Σ_i w_i
L_i(θ_i^E,θ_i^G) : le module est le bloc BCD, pas chaque sous-réseau).
"""
import torch

from .config import CONFIG
from .logger import log, log_verbeux
from .module import Module
from .utils import ajuster_dim


class ModuleVisuel(Module):
    def __init__(self, id, n_inputs_reco=None, n_latent=None, resolution=None, **kwargs):
        self.resolution = resolution or (CONFIG["n_frames"], CONFIG["taille_perception"],
                                         CONFIG["taille_perception"])
        c, h, w = self.resolution
        n_inputs_reco = n_inputs_reco or c * h * w
        n_latent = n_latent or CONFIG["dim_emb"]
        kwargs.setdefault("n_outputs_gen", c * h * w)
        super().__init__(id, n_inputs_reco, n_latent, **kwargs)

    # ------------------------------------------------------------------ poids
    def _init_poids(self):
        c, h, w = self.resolution
        canaux = CONFIG["conv_canaux"]
        self.enc_conv1 = torch.nn.Conv2d(c, canaux, 3, padding=1)
        self.enc_conv2 = torch.nn.Conv2d(canaux, canaux, 3, stride=2, padding=1)
        h2, w2 = (h + 1) // 2, (w + 1) // 2
        self._h2, self._w2, self._canaux = h2, w2, canaux
        self.enc_lin = torch.nn.Linear(canaux * h2 * w2, self.n_latent)

        self.dec_lin = torch.nn.Linear(self.n_latent, canaux * h2 * w2)
        self.dec_deconv = torch.nn.ConvTranspose2d(
            canaux, canaux, 3, stride=2, padding=1,
            output_padding=(h - (h2 - 1) * 2 - 3 + 2, w - (w2 - 1) * 2 - 3 + 2))
        self.dec_conv_sortie = torch.nn.Conv2d(canaux, c, 3, padding=1)
        self.reinj_lin = torch.nn.Linear(self.n_latent, self.dim_reinjection)

    def _rebuild_accumulateurs(self, voie=None):
        if voie in (None, "reco"):
            self._g_reco = [torch.zeros_like(p) for p in self.parametres_reco()]
        if voie in (None, "gen"):
            self._g_gen = [torch.zeros_like(p) for p in self.parametres_gen()]

    def parametres_reco(self):
        return (list(self.enc_conv1.parameters()) + list(self.enc_conv2.parameters())
                + list(self.enc_lin.parameters()))

    def parametres_gen(self):
        return (list(self.dec_lin.parameters()) + list(self.dec_deconv.parameters())
                + list(self.dec_conv_sortie.parameters()) + list(self.reinj_lin.parameters()))

    def grandir(self, voie, pas=None):
        """Croissance dimensionnelle non supportée pour l'architecture
        convolutive dans ce POC (idem ancien module_conv/module_slots)."""
        log(self.id, "croissance_non_supportee_visuel", voie=voie)
        return False

    # -------------------------------------------------------------- forward
    def forward_reconnaissance(self, input_niveau_inferieur):
        c, h, w = self.resolution
        x = ajuster_dim(input_niveau_inferieur, c * h * w).view(1, c, h, w)
        z = torch.relu(self.enc_conv1(x))
        z = torch.relu(self.enc_conv2(z))
        latent = self.enc_lin(z.reshape(-1))
        self.dernier_latent = latent.detach()
        log_verbeux(self.id, "forward_reconnaissance_visuel",
                    norme_latent=float(latent.detach().norm()))
        return latent

    def forward_generation(self, latent):
        z = ajuster_dim(latent, self.n_latent)
        h_dec = torch.relu(self.dec_lin(z)).view(1, self._canaux, self._h2, self._w2)
        h_dec = torch.relu(self.dec_deconv(h_dec))
        c, h, w = self.resolution
        sortie_visuelle = self.dec_conv_sortie(h_dec)[:, :, :h, :w].reshape(-1)
        reinj = self.reinj_lin(z)
        sortie_complete = torch.cat([sortie_visuelle, reinj])
        self.dernier_reinjecte = reinj.detach()
        log_verbeux(self.id, "forward_generation_visuel",
                    norme_output=float(sortie_visuelle.detach().norm()))
        return sortie_complete

    # --------------------------------------------- auto-supervision (JEPA)
    def entrainer_masque(self, champ_visuel, fraction_masque=0.5, phase="jour", t=0):
        """Masque aléatoire en entrée, cible = champ COMPLET propre.
        Entraînement conjoint encodeur+décodeur (une seule perte, un seul
        passage arrière — intra-module, autorisé par §0)."""
        c, h, w = self.resolution
        champ = ajuster_dim(champ_visuel, c * h * w).detach()
        masque = (torch.rand(c * h * w) > fraction_masque).float()
        entree_masquee = champ * masque

        for p in self.parametres_reco() + self.parametres_gen():
            p.grad = None
        latent = self.forward_reconnaissance(entree_masquee)
        sortie = self.forward_generation(latent)
        sortie_utile = sortie[: self.n_outputs_gen]
        erreur = torch.mean((sortie_utile - champ) ** 2)
        erreur.backward()

        grads_reco = [p.grad for p in self.parametres_reco()]
        grads_gen = [p.grad for p in self.parametres_gen()]
        self.incorporer_gradient(grads_reco, "reco", phase=phase)
        self.incorporer_gradient(grads_gen, "gen", phase=phase)
        lr = CONFIG["lr_normal"] if self.condensateur_reco < 0.5 else CONFIG["lr_lent"]
        self._appliquer_accumulateur(self.parametres_reco(), self._g_reco, lr)
        self._appliquer_accumulateur(self.parametres_gen(), self._g_gen, lr)

        e = float(erreur.detach())
        self.tentatives_count += 1
        self.meilleure_erreur_reco = min(self.meilleure_erreur_reco, e)
        self.meilleure_erreur_gen = min(self.meilleure_erreur_gen, e)
        self._enregistrer_erreur(None, e, t)
        log(self.id, "entrainement_masque", erreur=e, fraction_masque=fraction_masque,
            phase=phase)
        return e

    # ------------------------------------------- latent prédictif, multi-init
    def chercher_latent_predictif(self, input_prec, cible_future,
                                  n_iterations=None, n_inits=3):
        """Comme `Module`, mais avec plusieurs initialisations — l'espace
        latent visuel est potentiellement non convexe — garde la meilleure."""
        n_iterations = n_iterations or CONFIG["n_iterations_latent_predictif"]
        cible = ajuster_dim(cible_future, self.n_outputs_gen).detach()
        etats = [(p, p.requires_grad) for p in self.parametres_gen()]
        for p, _ in etats:
            p.requires_grad_(False)
        with torch.no_grad():
            z0 = self.forward_reconnaissance(
                ajuster_dim(input_prec, self.n_inputs_reco).detach())
        meilleur_z, meilleure_erreur = None, float("inf")
        for i in range(n_inits):
            bruit = torch.randn_like(z0) * (0.0 if i == 0 else 0.3)
            z = (z0 + bruit).detach().requires_grad_(True)
            for _ in range(n_iterations):
                sortie = self.forward_generation(z)[: self.n_outputs_gen]
                ecart = torch.mean((sortie - cible) ** 2)
                (grad,) = torch.autograd.grad(ecart, z)
                z = (z - CONFIG["lr_recherche_latent"] * grad).detach().requires_grad_(True)
            with torch.no_grad():
                e = float(torch.mean(
                    (self.forward_generation(z)[: self.n_outputs_gen] - cible) ** 2))
            if e < meilleure_erreur:
                meilleure_erreur, meilleur_z = e, z.detach()
        for p, s in etats:
            p.requires_grad_(s)
        log_verbeux(self.id, "latent_predictif_visuel", ecart_final=meilleure_erreur,
                    n_inits=n_inits)
        return meilleur_z
