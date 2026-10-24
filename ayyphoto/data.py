"""Index local des images et paires synthétiques reproductibles."""

from collections import Counter, OrderedDict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .imaging import decode_photo


@dataclass(frozen=True)
class Record:
    path: str
    split: str
    scene: str


def index_images(root, seed=42):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dossier de données introuvable : {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not files:
        raise ValueError("Ce dossier ne contient aucune image JPEG/PNG.")
    official = any("DIV2K_train_HR" in path.relative_to(root).parts for path in files)
    records = []
    if official:
        train_ids = {f"{i:04d}" for i in range(1, 801)}
        test_ids = {f"{i:04d}" for i in range(801, 901)}
        val_ids = set(random.Random(seed).sample(sorted(train_ids), 80))
        seen = Counter()
        for path in files:
            relative = path.relative_to(root)
            seen[path.stem] += 1
            if "DIV2K_train_HR" in relative.parts[:-1] and path.stem in train_ids:
                split = "val" if path.stem in val_ids else "train"
            elif "DIV2K_valid_HR" in relative.parts[:-1] and path.stem in test_ids:
                split = "test"
            else:
                raise ValueError(f"Image hors du contrat DIV2K : {relative}")
            records.append(Record(relative.as_posix(), split, f"div2k_{path.stem}"))
        if set(seen) != train_ids | test_ids or any(count != 1 for count in seen.values()):
            raise ValueError("Le dataset DIV2K doit contenir les 900 identifiants uniques 0001 à 0900.")
    else:
        scenes = {}
        for path in files:
            relative = path.relative_to(root)
            if len(relative.parts) < 3 or relative.parts[0] not in {"train", "val", "test"}:
                raise ValueError("Pour un dataset personnel, utiliser train/scene_001/photo.png et val/scene_101/photo.png.")
            split, scene = relative.parts[:2]
            if scene in scenes and scenes[scene] != split:
                raise ValueError(f"Scène commune entre les ensembles : {scene}")
            scenes[scene] = split
            records.append(Record(relative.as_posix(), split, scene))
    if not any(row.split == "train" for row in records) or not any(row.split == "val" for row in records):
        raise ValueError("Il faut au moins une image d’entraînement et une image de validation indépendante.")
    return records


def fingerprint(root, records):
    digest = hashlib.sha256()
    seen = {}
    for record in records:
        digest.update(f"{record.path},{record.split},{record.scene}\n".encode())
        file_digest = hashlib.sha256()
        with (Path(root) / record.path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                file_digest.update(chunk)
        checksum = file_digest.hexdigest()
        previous = seen.get(checksum)
        if previous and previous.split != record.split:
            raise ValueError(f"Doublon exact entre ensembles : {previous.path} et {record.path}. Revoir la séparation des données.")
        seen[checksum] = record
        digest.update(checksum.encode())
    return digest.hexdigest()


def select_records(records, split, limit, seed):
    rows = [record for record in records if record.split == split]
    if limit and len(rows) > limit:
        selected = set(random.Random(seed).sample(range(len(rows)), limit))
        rows = [record for index, record in enumerate(rows) if index in selected]
    return rows


def check_dimensions(root, records, minimum):
    for record in records:
        with Image.open(Path(root) / record.path) as image:
            if min(image.size) < minimum:
                raise ValueError(f"Image trop petite pour un recadrage {minimum}×{minimum} : {record.path}")


class SyntheticPairs(Dataset):
    def __init__(self, root, records, config, training=False):
        self.root, self.records, self.config = Path(root), list(records), config
        self.training, self.epoch = training, 0
        self.cache, self.cache_bytes = OrderedDict(), 0
        self.cache_limit = config.rgb_cache_mib * 2**20

    def __len__(self):
        repeats = self.config.patches_per_image if self.training else len(self.config.eval_sigmas)
        return len(self.records) * repeats

    def _read(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        rgb = decode_photo((self.root / path).read_bytes())
        if rgb.nbytes <= self.cache_limit:
            while self.cache and self.cache_bytes + rgb.nbytes > self.cache_limit:
                _, removed = self.cache.popitem(last=False)
                self.cache_bytes -= removed.nbytes
            self.cache[path], self.cache_bytes = rgb, self.cache_bytes + rgb.nbytes
        return rgb

    def __getitem__(self, index):
        cfg = self.config
        repeats = cfg.patches_per_image if self.training else len(cfg.eval_sigmas)
        record = self.records[index // repeats]
        rng = np.random.default_rng(np.random.SeedSequence(
            [cfg.seed, self.epoch, index, 1] if self.training else [cfg.seed, index, 2]))
        clean = self._read(record.path)
        size = cfg.patch_size if self.training else cfg.eval_size
        height, width = clean.shape[:2]
        top = int(rng.integers(height-size+1)) if self.training else (height-size)//2
        left = int(rng.integers(width-size+1)) if self.training else (width-size)//2
        clean = clean[top:top+size, left:left+size]
        if self.training:
            clean = np.rot90(clean, int(rng.integers(4)))
            if rng.random() < 0.5:
                clean = np.fliplr(clean)
            sigma = float(rng.uniform(cfg.sigma_min, cfg.sigma_max))
        else:
            sigma = cfg.eval_sigmas[index % repeats]
        clean = clean.astype(np.float32) / 255.0
        noise = rng.normal(0, sigma/255.0, clean.shape).astype(np.float32)
        noisy = np.clip(clean+noise, 0, 1)
        to_tensor = lambda value: torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))
        return to_tensor(noisy), to_tensor(clean)
