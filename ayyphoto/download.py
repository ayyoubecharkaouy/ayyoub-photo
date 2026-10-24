"""Préparer DIV2K en local : vérifier, télécharger si nécessaire, puis vérifier."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

from .config import PROJECT_ROOT, write_json


DATASET_HANDLE = "soumikrakshit/div2k-high-resolution-images"
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "archive"
EXPECTED = {f"{index:04d}": "DIV2K_train_HR" if index <= 800 else "DIV2K_valid_HR"
            for index in range(1, 901)}
REMOTE_PATHS = {key: f"{folder}/{folder}/{key}.png" for key, folder in EXPECTED.items()}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class DatasetStatus:
    root: Path
    expected: dict
    paths: dict = field(default_factory=dict)
    valid: dict = field(default_factory=dict)
    damaged: dict = field(default_factory=dict)
    deep: bool = False

    @property
    def missing(self):
        return sorted(set(self.expected) - set(self.paths))

    @property
    def complete(self):
        return len(self.valid) == len(self.expected) and not self.damaged and not self.missing

    def summary(self):
        return {
            "destination": str(self.root), "expected_images": len(self.expected),
            "present_images": len(self.paths), "valid_images": len(self.valid),
            "missing_images": len(self.missing), "damaged_images": len(self.damaged),
            "complete": self.complete, "verification": "full_decode" if self.deep else "inventory_only",
            "missing_examples": self.missing[:10],
            "damaged_examples": dict(list(self.damaged.items())[:10]),
        }


def _image_dependencies():
    try:
        from PIL import Image, ImageCms, ImageOps
        return Image, ImageCms, ImageOps
    except ImportError as exc:
        raise RuntimeError("Pillow est nécessaire. Installer requirements-data.txt dans .venv.") from exc


def verify_image(path):
    Image, ImageCms, ImageOps = _image_dependencies()
    raw = Path(path).read_bytes()
    if not raw or len(raw) > 32 * 1024 * 1024:
        raise ValueError("Fichier vide ou supérieur à 32 Mio.")
    with Image.open(io.BytesIO(raw)) as image:
        if image.format not in {"PNG", "JPEG"} or image.width * image.height > 16_000_000:
            raise ValueError("Format ou dimensions incompatibles avec le pipeline local.")
        if min(image.size) < 256:
            raise ValueError("Le petit côté doit mesurer au moins 256 pixels.")
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            raise ValueError("Image avec transparence.")
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        rgb = ImageOps.exif_transpose(image)
        profile = rgb.info.get("icc_profile")
        if profile:
            try:
                rgb = ImageCms.profileToProfile(rgb, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                                               ImageCms.createProfile("sRGB"), outputMode="RGB")
            except ImageCms.PyCMSError as exc:
                raise ValueError("Profil couleur invalide.") from exc
        else:
            rgb = rgb.convert("RGB")
        rgb.load()
        width, height = rgb.size
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "width": width, "height": height}


def inventory(root, expected=None):
    root = Path(root).expanduser().resolve()
    expected = EXPECTED if expected is None else expected
    result = DatasetStatus(root=root, expected=expected)
    if not root.exists():
        return result
    if not root.is_dir():
        raise ValueError("La destination doit être un dossier.")
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = path.relative_to(root)
        group = expected.get(path.stem)
        if group is None or group not in relative.parts[:-1]:
            raise ValueError(f"Ce dossier contient une image étrangère à DIV2K : {relative}. Choisir un dossier dédié.")
        if path.stem in result.paths:
            raise ValueError(f"L’identifiant {path.stem} apparaît plusieurs fois. Résoudre ce doublon avant de préparer les données.")
        result.paths[path.stem] = path
        if path.stat().st_size:
            result.valid[path.stem] = {"bytes": path.stat().st_size}
        else:
            result.damaged[path.stem] = "Fichier vide."
    return result


def inspect_dataset(root, expected=None, progress=None):
    result = inventory(root, expected)
    if result.paths:
        _image_dependencies()  # Une dépendance absente n'est pas une corruption des images.
    result.deep = True
    result.valid.clear()
    for index, (image_id, path) in enumerate(result.paths.items(), 1):
        if progress and (index % 20 == 0 or index == len(result.paths)):
            progress({"state": "verifying", "completed": index, "total": len(result.paths),
                      "message": f"Vérification des images présentes : {index}/{len(result.paths)}"})
        try:
            result.valid[image_id] = verify_image(path)
            result.damaged.pop(image_id, None)
        except (OSError, ValueError) as exc:
            result.damaged[image_id] = str(exc)
    return result


def kaggle_download(relative_path=None):
    # Cache distinct des images finales pour ne pas créer de doublons dans leur index.
    os.environ.setdefault("KAGGLEHUB_CACHE", str(PROJECT_ROOT / ".cache" / "kagglehub"))
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("KaggleHub manque. Installer requirements-data.txt dans .venv.") from exc
    arguments = {"path": relative_path} if relative_path is not None else {}
    try:
        return Path(kagglehub.dataset_download(DATASET_HANDLE, **arguments)).resolve()
    except Exception as exc:
        response = getattr(exc, "response", None)
        if relative_path is not None and (isinstance(exc, FileNotFoundError) or getattr(response, "status_code", None) == 404):
            raise FileNotFoundError(f"Chemin distant absent : {relative_path}") from exc
        raise RuntimeError(
            "Le téléchargement Kaggle a échoué. Vérifier Internet, l’espace disque et l’accès au dataset. "
            "Si Kaggle exige une connexion, utiliser kagglehub.login() dans le terminal de .venv."
        ) from exc


def prepare_dataset(destination=DEFAULT_DESTINATION, downloader=None, progress=None, expected=None):
    expected = EXPECTED if expected is None else expected
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.prepare.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Une préparation de ce dataset est déjà en cours.") from exc
        before = inspect_dataset(destination, expected, progress)
        if before.complete:
            report = {**before.summary(), "downloaded": False, "installed_files": 0,
                      "message": "Toutes les images sont valides : aucun téléchargement nécessaire."}
        else:
            _image_dependencies()
            download = kaggle_download if downloader is None else downloader
            needed = sorted(set(expected)-set(before.valid))
            if progress:
                progress({"state": "downloading", "message": f"Récupération de {len(needed)} images manquantes ou invalides…"})
            cached_root = None
            # Les petites réparations évitent de récupérer une archive complète.
            individual = len(needed) <= 32
            if not individual:
                cached_root = Path(download()).resolve()
                source_index = inventory(cached_root, expected)
                absent = set(needed)-set(source_index.paths)
                if absent:
                    raise ValueError(f"La source Kaggle ne contient pas les identifiants attendus : {sorted(absent)[:10]}")
            for index, image_id in enumerate(needed, 1):
                if individual and cached_root is None:
                    group = expected[image_id]
                    remote_path = REMOTE_PATHS.get(image_id, f"{group}/{group}/{image_id}.png")
                    try:
                        source = Path(download(remote_path)).resolve()
                    except FileNotFoundError:
                        # Une nouvelle version du dataset peut avoir changé son arborescence.
                        cached_root = Path(download()).resolve()
                        source_index = inventory(cached_root, expected)
                        if image_id not in source_index.paths:
                            raise ValueError(f"Identifiant absent du dataset téléchargé : {image_id}")
                        source = source_index.paths[image_id]
                    if source.is_dir():
                        source = inventory(source, expected).paths.get(image_id)
                        if source is None:
                            raise ValueError(f"Fichier demandé absent de la source Kaggle : {image_id}")
                else:
                    if image_id not in source_index.paths:
                        raise ValueError(f"Identifiant absent du dataset téléchargé : {image_id}")
                    source = source_index.paths[image_id]
                metadata = verify_image(source)
                output = before.paths.get(image_id, destination / expected[image_id] / f"{image_id}.png")
                output.parent.mkdir(parents=True, exist_ok=True)
                if shutil.disk_usage(output.parent).free < metadata["bytes"] + 64*1024*1024:
                    raise OSError("Espace disque insuffisant pour terminer la copie du dataset.")
                temporary = output.with_name(output.name + ".ayyphoto-partial")
                shutil.copyfile(source, temporary)
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != metadata["sha256"]:
                    raise OSError(f"La copie de {image_id} n’est pas complète ; relancer la préparation.")
                if output.exists():
                    # Conserver le fichier abîmé avant remplacement, sans l'exposer au chargeur d'images.
                    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
                    output.replace(output.with_name(output.name + f".invalid_{stamp}"))
                temporary.replace(output)
                if progress:
                    progress({"state": "downloading", "completed": index, "total": len(needed),
                              "message": f"Images installées : {index}/{len(needed)}"})
            after = inspect_dataset(destination, expected, progress)
            if not after.complete:
                raise ValueError("La vérification finale du dataset a échoué ; les fichiers valides sont conservés.")
            before = after
            report = {**after.summary(), "downloaded": True, "installed_files": len(needed),
                      "message": "Dataset téléchargé, vérifié et prêt pour le chargeur local."}
        metadata_path = destination.parent / f".{destination.name}.div2k.json"
        write_json(metadata_path, {
            "handle": DATASET_HANDLE, "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "report": report,
            "files": {str(before.paths[key].relative_to(destination)): metadata for key, metadata in before.valid.items()},
        })
        return report


def prepare_training_data(config, directory):
    if not config.prepare_div2k:
        return
    directory = Path(directory)

    def progress(value):
        print(value["message"], flush=True)
        write_json(directory / "status.json", value)

    progress({"state": "verifying", "message": "Vérification des données avant l’entraînement…"})
    report = prepare_dataset(config.data_root, progress=progress)
    if not report["complete"]:
        raise ValueError("Préparation des données incomplète ; entraînement interrompu.")
    write_json(directory / "dataset_report.json", report)
    progress({"state": "starting", "message": "Données prêtes. Préparation de l’entraînement CPU…"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--check", action="store_true", help="Vérifier entièrement les images sans accès réseau.")
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    directory = args.config.resolve().parent if args.config else None

    def progress(value):
        print(value["message"], flush=True)
        if directory:
            write_json(directory / "status.json", value)

    try:
        if args.config:
            payload = json.loads(args.config.read_text(encoding="utf-8"))
            args.destination = Path(payload["data_root"])
        if args.check:
            report = inspect_dataset(args.destination, progress=progress).summary()
        else:
            report = prepare_dataset(args.destination, progress=progress)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if directory:
            write_json(directory / "dataset_report.json", report)
            write_json(directory / "status.json", {"state": "completed" if report["complete"] else "error",
                                                   "message": report.get("message", "Vérification des images terminée."),
                                                   "valid_images": report["valid_images"]})
        return 0 if report["complete"] else 2
    except KeyboardInterrupt:
        progress({"state": "cancelled", "message": "Préparation arrêtée ; les images déjà installées sont conservées."})
        return 130
    except Exception as exc:
        progress({"state": "error", "message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
