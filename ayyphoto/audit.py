"""Exécuter l'audit de données local comme calcul suivi par Streamlit."""

import argparse
import json
from pathlib import Path
import traceback

from ml.data_audit import audit_dataset
from .config import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    directory = args.config.resolve().parent
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))

        def progress(files, desc):
            for index, file in enumerate(files):
                if index % 10 == 0:
                    write_json(directory / "status.json", {"state": "running", "completed": index,
                        "total": len(files), "message": f"Lecture des images : {index}/{len(files)}"})
                yield file

        summary = audit_dataset(config["data_root"], directory / "audit", progress=progress)
        write_json(directory / "status.json", {"state": "completed", "message": "Audit terminé.",
                                               "decoded_images": summary["decoded_images"], "errors_count": summary["errors_count"]})
        return 0
    except KeyboardInterrupt:
        write_json(directory / "status.json", {"state": "cancelled", "message": "Audit arrêté ; les images sources sont inchangées."})
        return 130
    except Exception as exc:
        write_json(directory / "status.json", {"state": "error", "message": str(exc)})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
