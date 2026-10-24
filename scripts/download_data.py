#!/usr/bin/env python3
"""Vérifier et préparer les images DIV2K dans data/archive."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ayyphoto.download import main


if __name__ == "__main__":
    raise SystemExit(main())
