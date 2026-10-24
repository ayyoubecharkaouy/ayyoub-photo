"""Configuration locale et écritures atomiques sans dépendances ML."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT_ROOT / "experiments" / "local"


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


@dataclass
class TrainConfig:
    data_root: str
    epochs: int = 2
    batch_size: int = 1
    patch_size: int = 64
    eval_size: int = 128
    patches_per_image: int = 2
    max_train_images: int = 40  # 0 = toutes les images.
    max_val_images: int = 8
    channels: int = 48
    depth: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    sigma_min: float = 0.0
    sigma_max: float = 50.0
    eval_sigmas: tuple = (15.0, 25.0, 50.0)
    patience: int = 8
    seed: int = 42
    threads: int = 2
    rgb_cache_mib: int = 16
    initial_model: str = ""
    resume_run: str = ""
    prepare_div2k: bool = False

    def validate(self):
        root = Path(self.data_root).expanduser().resolve()
        if not isinstance(self.prepare_div2k, bool):
            raise ValueError("La préparation automatique doit être activée ou désactivée.")
        if root.exists() and not root.is_dir():
            raise ValueError(f"Le chemin des images doit être un dossier : {root}")
        if not root.is_dir() and not self.prepare_div2k:
            raise ValueError(f"Dossier d’images introuvable : {root}")
        if any(value < 1 for value in (self.epochs, self.batch_size, self.patches_per_image,
                                       self.channels, self.patience, self.threads)):
            raise ValueError("Époques, batch, recadrages, canaux, patience et threads doivent être positifs.")
        if self.depth < 2 or self.patch_size < 16 or self.eval_size < self.patch_size:
            raise ValueError("Profondeur ≥ 2, recadrage ≥ 16 et taille de validation ≥ recadrage requis.")
        if self.max_train_images < 0 or self.max_val_images < 0 or self.rgb_cache_mib < 0:
            raise ValueError("Les limites d’images et de cache doivent être positives ou nulles.")
        if not 0 < self.learning_rate <= 1 or self.weight_decay < 0:
            raise ValueError("Taux d’apprentissage ou régularisation invalide.")
        if not 0 <= self.sigma_min < self.sigma_max <= 255:
            raise ValueError("Les niveaux de bruit doivent respecter 0 ≤ min < max ≤ 255.")
        if not self.eval_sigmas or not all(0 < sigma <= self.sigma_max for sigma in self.eval_sigmas):
            raise ValueError("Niveaux de bruit de validation invalides.")
        if self.initial_model and self.resume_run:
            raise ValueError("Choisir soit des poids initiaux, soit une reprise d’expérience.")
        for value in (self.initial_model,):
            if value and not Path(value).expanduser().is_file():
                raise ValueError(f"Fichier de poids introuvable : {value}")
        if self.resume_run and not Path(self.resume_run).expanduser().is_dir():
            raise ValueError("Le dossier de reprise n’existe pas.")
        self.data_root = str(root)
        return self

    @classmethod
    def from_dict(cls, value):
        allowed = cls.__dataclass_fields__
        unknown = set(value) - set(allowed)
        if unknown:
            raise ValueError(f"Options non reconnues : {sorted(unknown)}")
        return cls(**value)

    def to_dict(self):
        return asdict(self)
