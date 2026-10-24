#!/usr/bin/env python3
"""Démarrer Streamlit avec le Python de .venv et les chemins locaux du projet."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def main():
    if not PYTHON.is_file():
        print("Exécuter d’abord : python3 scripts/setup_local.py", file=sys.stderr)
        return 1
    result = subprocess.run([str(PYTHON), "-c", "import torch, streamlit, numpy, PIL, skimage"],
                            capture_output=True, text=True)
    if result.returncode:
        print("Installation incomplète. Exécuter : python3 scripts/setup_local.py", file=sys.stderr)
        return 1
    os.chdir(ROOT)
    os.execv(str(PYTHON), [str(PYTHON), "-m", "streamlit", "run", str(ROOT / "app.py"),
                         "--server.address=127.0.0.1", "--server.port=8501"])


if __name__ == "__main__":
    raise SystemExit(main())
