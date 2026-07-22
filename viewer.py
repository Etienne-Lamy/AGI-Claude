#!/usr/bin/env python3
"""Serveur du dashboard SCL temps réel.

Suit scl_audit.jsonl pendant que run_poc.py tourne et sert viewer.html.

Usage (dans un 2e terminal, pendant le run) :
    python3 viewer.py --log scl_audit.jsonl --port 8400
puis ouvrir http://localhost:8400 dans le navigateur.
"""
import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Actions pertinentes pour le dashboard (le reste du log est trop verbeux).
# Vocabulaire v7 (dashboard « modules / graphe / champ ») — produit par
# scl/demo_viewer.py et les détecteurs/orchestrateurs des étapes 10-15.
ACTIONS_UTILES = {
    # méta + rythme des phases
    "meta", "phase", "programme_choisi",
    # champ VU vs PRÉVU + état des modules au fil du temps
    "champ", "modules_etat",
    # agent objet (perçoit → prédit → planifie A* → agit) — étape 28
    "agent",
    # cycle de vie des modules-régime
    "naissance_module_regime", "verrouillage_module_regime",
    # graphe de branchement de l'orchestrateur (Mode A)
    "programme_evalue",
    # rétro-compat POC v6 (si on rejoue un vieux log run_poc.py)
    "pouls", "creation_predicteur", "resume_journee",
}
MAX_LIGNES_PAR_REQUETE = 8000


class Handler(BaseHTTPRequestHandler):
    chemin_log = "scl_audit.jsonl"
    chemin_html = "viewer.html"

    def log_message(self, *a):  # silence du serveur
        pass

    def _repondre(self, code, contenu, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(contenu)))
        self.end_headers()
        self.wfile.write(contenu)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            try:
                with open(self.chemin_html, "rb") as f:
                    self._repondre(200, f.read(), "text/html")
            except FileNotFoundError:
                self._repondre(404, b"viewer.html introuvable", "text/plain")
        elif url.path == "/data":
            offset = int(parse_qs(url.query).get("offset", ["0"])[0])
            self._repondre(200, json.dumps(self._lire(offset),
                                           ensure_ascii=False).encode("utf-8"))
        else:
            self._repondre(404, b"?", "text/plain")

    def _lire(self, offset):
        """Lit les nouvelles lignes complètes du JSONL depuis offset (octets)."""
        lignes = []
        try:
            taille = os.path.getsize(self.chemin_log)
        except OSError:
            return {"offset": 0, "lines": [], "eof": True}
        if offset > taille:   # fichier recréé (nouveau run) : repartir de zéro
            offset = 0
        with open(self.chemin_log, "rb") as f:
            f.seek(offset)
            data = f.read()
        fin_derniere_ligne = data.rfind(b"\n")
        if fin_derniere_ligne == -1:
            return {"offset": offset, "lines": [], "eof": True}
        data = data[: fin_derniere_ligne + 1]
        nouveau_offset = offset + len(data)
        for brut in data.split(b"\n"):
            if not brut.strip():
                continue
            try:
                # parse_constant : NaN/Infinity (tolérés par Python, invalides
                # en JSON standard) deviennent null au lieu de tuer JSON.parse
                # côté navigateur — robustesse aux anciens logs
                rec = json.loads(brut, parse_constant=lambda _: None)
            except json.JSONDecodeError:
                continue
            if rec.get("action") in ACTIONS_UTILES:
                lignes.append(rec)
            if len(lignes) >= MAX_LIGNES_PAR_REQUETE:
                break
        return {"offset": nouveau_offset, "lines": lignes,
                "eof": nouveau_offset >= taille}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", default="scl_audit.jsonl")
    p.add_argument("--port", type=int, default=8400)
    args = p.parse_args()
    Handler.chemin_log = args.log
    Handler.chemin_html = os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "viewer.html")
    serveur = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard SCL : http://localhost:{args.port}  (log : {args.log})")
    try:
        serveur.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
