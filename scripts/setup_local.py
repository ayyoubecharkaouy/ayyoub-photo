#!/usr/bin/env python3
"""Créer .venv et installer les dépendances CPU d'AyyPhoto."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import venv


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def check_environment():
    if not PYTHON.exists():
        print("Environnement .venv absent. Exécuter python3 scripts/setup_local.py.")
        return 1
    code = '''import importlib, json, sys
packages = ["pip", "torch", "streamlit", "numpy", "PIL", "skimage", "kagglehub"]
found, errors = {}, {}
for name in packages:
    try:
        importlib.import_module(name)
        found[name] = True
    except Exception as exc:
        found[name] = False
        errors[name] = str(exc)
print(json.dumps({"python": sys.executable, "packages": found, "errors": errors}, indent=2))
if all(found.values()):
    import torch
    from ayyphoto.runtime import AyyPhotoDenoiser
    torch.set_num_threads(2)
    with torch.inference_mode():
        assert AyyPhotoDenoiser(channels=4, depth=2)(torch.zeros(1,3,17,19)).shape == (1,3,17,19)
    print("Forward CPU vérifié.")
raise SystemExit(0 if all(found.values()) else 1)
'''
    return subprocess.run([str(PYTHON), "-c", code], cwd=ROOT).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Vérifier l’environnement sans installer.")
    parser.add_argument("--prepare-only", action="store_true", help="Créer .venv et pip sans télécharger de dépendances.")
    args = parser.parse_args()
    if args.check:
        return check_environment()
    if sys.version_info < (3, 10):
        print("Python 3.10 ou plus récent est requis.", file=sys.stderr)
        return 1
    try:
        print("Préparation de .venv…", flush=True)
        venv.EnvBuilder(with_pip=True, symlinks=True).create(ROOT / ".venv")
        if args.prepare_only:
            print(".venv et pip sont prêts ; aucune dépendance téléchargée.")
            return 0
        environment = os.environ.copy()
        environment["PIP_CACHE_DIR"] = str(ROOT / ".cache" / "pip")
        for filename in ("requirements-torch-cpu.txt", "requirements.txt"):
            subprocess.run([str(PYTHON), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(ROOT / filename)],
                           cwd=ROOT, env=environment, check=True)
        print("Installation terminée. Lancer : python3 scripts/run_local.py")
        return check_environment()
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Installation non terminée : {exc}", file=sys.stderr)
        print("Vérifier la connexion réseau et relancer cette commande ; aucun paquet système n’est modifié.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
