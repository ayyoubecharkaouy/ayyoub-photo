"""CNN compatible avec les notebooks, chargement des poids et inférence CPU."""

from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
import re
from threading import RLock
import zipfile

import numpy as np
import torch
from torch import nn


MAX_MODEL_BYTES = 64 * 1024 * 1024
MODEL_NAMES = ("model_bundle.pt", "best_weights.pth", "last.ckpt", "interrupted_weights.pth")


class AyyPhotoDenoiser(nn.Module):
    def __init__(self, channels=48, depth=8):
        super().__init__()
        layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, 3, 3, padding=1)]
        self.network = nn.Sequential(*layers)
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, image):
        return image - self.network(image)


def infer_architecture(weights):
    indices = sorted(int(match.group(1)) for key in weights
                     if (match := re.fullmatch(r"network\.(\d+)\.weight", key)))
    depth = len(indices)
    if depth < 2 or indices != list(range(0, depth * 2, 2)):
        raise ValueError("Ces poids ne correspondent pas au CNN AyyPhoto des notebooks.")
    channels = int(weights["network.0.weight"].shape[0])
    if channels < 1:
        raise ValueError("Nombre de canaux invalide.")
    for position, index in enumerate(indices):
        expected = (channels, 3, 3, 3) if position == 0 else (
            (3, channels, 3, 3) if position == depth - 1 else (channels, channels, 3, 3))
        if tuple(weights[f"network.{index}.weight"].shape) != expected:
            raise ValueError(f"Architecture incompatible à la couche {index}.")
    return {"channels": channels, "depth": depth}


@dataclass
class LoadedModel:
    model: AyyPhotoDenoiser
    architecture: dict
    sha256: str
    label: str
    lock: RLock = field(default_factory=RLock)


def load_model(raw, label="modèle", threads=2):
    if not raw or len(raw) > MAX_MODEL_BYTES:
        raise ValueError("Choisir un modèle AyyPhoto de moins de 64 Mio.")
    content = raw
    source = label
    stream = io.BytesIO(raw)
    # Les fichiers PyTorch sont aussi des ZIP : seul un ZIP contenant nos noms
    # d'artefacts est interprété comme archive d'export, sans extraction sur disque.
    if zipfile.is_zipfile(stream):
        try:
            with zipfile.ZipFile(stream) as archive:
                entries = [entry for entry in archive.infolist()
                           if not entry.is_dir() and Path(entry.filename).name in MODEL_NAMES]
                if entries:
                    rank = min(MODEL_NAMES.index(Path(entry.filename).name) for entry in entries)
                    selected = [entry for entry in entries if MODEL_NAMES.index(Path(entry.filename).name) == rank]
                    if len(selected) != 1:
                        raise ValueError("Plusieurs modèles dans cette archive ; choisir les poids d’une seule expérience.")
                    entry = selected[0]
                    if entry.file_size > MAX_MODEL_BYTES:
                        raise ValueError("Les poids contenus dans l’archive dépassent 64 Mio.")
                    content = archive.read(entry)
                    source = f"{label} — {entry.filename}"
        except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
            raise ValueError("L’archive de modèle est endommagée ou ne peut pas être ouverte.") from exc
    try:
        payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Impossible de charger ces poids. Choisir un fichier ou un export AyyPhoto valide.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Format de modèle non reconnu.")
    if payload.get("architecture", "residual_rgb_cnn") != "residual_rgb_cnn":
        raise ValueError("Architecture non prise en charge.")
    weights = payload.get("model_state_dict", payload)
    if not isinstance(weights, dict) or not weights or not all(isinstance(key, str) for key in weights):
        raise ValueError("Le fichier ne contient pas de poids.")
    if all(key.startswith("module.") for key in weights):
        weights = {key.removeprefix("module."): value for key, value in weights.items()}
    try:
        architecture = infer_architecture(weights)
        for key, value in weights.items():
            if not torch.is_tensor(value) or not torch.isfinite(value).all():
                raise ValueError(f"Poids non finis ou invalides : {key}")
        metadata = payload.get("config", {})
        if any(key in metadata and metadata[key] != architecture[key] for key in architecture):
            raise ValueError("Les métadonnées ne correspondent pas aux couches sauvegardées.")
        torch.set_num_threads(max(1, int(threads)))
        model = AyyPhotoDenoiser(**architecture)
        model.load_state_dict(weights, strict=True)
        model.eval()
    except (KeyError, TypeError, RuntimeError, AttributeError) as exc:
        raise ValueError("Les couches ou les tenseurs ne correspondent pas au modèle AyyPhoto.") from exc
    return LoadedModel(model, architecture, hashlib.sha256(content).hexdigest(), source)


def load_model_path(path, threads=2):
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.stat().st_size > MAX_MODEL_BYTES:
        raise ValueError("Fichier de poids absent ou trop volumineux.")
    return load_model(path.read_bytes(), path.name, threads)


def discover_local_models(roots):
    found = []
    for root in roots:
        root = Path(root)
        if root.is_dir():
            for pattern in (*MODEL_NAMES, "*_export.zip"):
                found.extend(path.resolve() for path in root.rglob(pattern) if path.is_file())
    return sorted(set(found), key=str)


@torch.inference_mode()
def denoise(loaded, rgb, strength=1.0, tile_size=128, progress=None):
    if not 0 <= strength <= 1 or tile_size < 16:
        raise ValueError("Intensité ou taille de tuile invalide.")
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Une photo RGB uint8 est requise.")
    if strength == 0:
        return rgb.copy()
    height, width = rgb.shape[:2]
    halo = max(16, loaded.architecture["depth"])
    # Garder la photo complète en uint8 ; convertir seulement chaque tuile.
    output = np.empty_like(rgb)
    total = ((height + tile_size - 1) // tile_size) * ((width + tile_size - 1) // tile_size)
    completed = 0
    with loaded.lock:
        for top in range(0, height, tile_size):
            bottom = min(top + tile_size, height)
            for left in range(0, width, tile_size):
                right = min(left + tile_size, width)
                y0, y1 = max(0, top - halo), min(height, bottom + halo)
                x0, x1 = max(0, left - halo), min(width, right + halo)
                array = np.ascontiguousarray(rgb[y0:y1, x0:x1].transpose(2, 0, 1))
                patch = torch.from_numpy(array).float().unsqueeze(0) / 255.0
                restored = loaded.model(patch)[0]
                if not torch.isfinite(restored).all():
                    raise FloatingPointError("Le modèle a produit des valeurs non finies.")
                restored = restored.clamp(0, 1)
                core = restored[:, top-y0:bottom-y0, left-x0:right-x0].permute(1, 2, 0).numpy()
                if not np.isfinite(core).all():
                    raise FloatingPointError("Le modèle a produit des valeurs non finies.")
                original = rgb[top:bottom, left:right].astype(np.float32) / 255.0
                output[top:bottom, left:right] = np.rint((original * (1-strength) + core * strength).clip(0, 1) * 255).astype(np.uint8)
                completed += 1
                if progress:
                    progress(completed / total)
    return output
