"""Audit CPU des photos DIV2K, sans entraînement et sans modification des sources."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import random
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageCms, ImageDraw, ImageOps


EXTENSIONS = {".png", ".jpg", ".jpeg"}
SEED = 42


def identify_split(relative_path, seed=SEED):
    path = Path(relative_path)
    parents = {part.lower() for part in path.parts[:-1]}
    image_id = path.stem
    validation_ids = set(random.Random(seed).sample([f"{i:04d}" for i in range(1, 801)], 80))
    if "div2k_train_hr" in parents and image_id in {f"{i:04d}" for i in range(1, 801)}:
        return "val" if image_id in validation_ids else "train"
    if "div2k_valid_hr" in parents and image_id in {f"{i:04d}" for i in range(801, 901)}:
        return "test"
    return "unknown"


def decode_image(raw):
    # verify contrôle aussi les sommes de contrôle PNG, puis une seconde lecture
    # décode tous les pixels à partir des mêmes octets en mémoire.
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        info = {
            "source_mode": image.mode,
            "has_icc": bool(image.info.get("icc_profile")),
            "has_gps_metadata": bool(image.getexif().get(34853)),
        }
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            raise ValueError("Transparence non acceptée par le pipeline de photos RGB.")
        rgb = ImageOps.exif_transpose(image)
        profile = rgb.info.get("icc_profile")
        if profile:
            rgb = ImageCms.profileToProfile(
                rgb, ImageCms.ImageCmsProfile(io.BytesIO(profile)),
                ImageCms.createProfile("sRGB"), outputMode="RGB",
            )
        else:
            rgb = rgb.convert("RGB")
        rgb.load()
        return rgb.copy(), info


def image_features(rgb):
    pixels = np.asarray(rgb, dtype=np.uint8)
    gray = np.asarray(rgb.convert("L"), dtype=np.uint8)
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / histogram.sum()
    nonzero = probabilities[probabilities > 0]
    thumbnail = rgb.resize((48, 48), Image.Resampling.LANCZOS)
    tiny = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    positions = np.arange(32)
    frequencies = np.arange(8)[:, None]
    basis = np.cos(np.pi * (2 * positions + 1) * frequencies / 64) * np.sqrt(2 / 32)
    basis[0] /= np.sqrt(2)
    low_frequencies = (basis @ tiny @ basis.T).ravel()[1:]
    phash = sum(int(bit) << index for index, bit in enumerate(low_frequencies > np.median(low_frequencies)))
    sharpness_image = np.asarray(rgb.convert("L").resize((256, 256), Image.Resampling.LANCZOS), dtype=np.float32) / 255.0
    laplacian = (sharpness_image[1:-1, :-2] + sharpness_image[1:-1, 2:]
                 + sharpness_image[:-2, 1:-1] + sharpness_image[2:, 1:-1]
                 - 4 * sharpness_image[1:-1, 1:-1])
    result = {
        "width": rgb.width, "height": rgb.height,
        "pixel_sha256": hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest(),
        "mean_luminance": float(np.dot(probabilities, np.arange(256)) / 255),
        "luminance_std": float(gray.std() / 255),
        "dark_fraction": float(probabilities[:6].sum()),
        "bright_fraction": float(probabilities[250:].sum()),
        "entropy_bits": float(-(nonzero * np.log2(nonzero)).sum()),
        "laplacian_variance_256": float(laplacian.var()),
        "phash63": f"{phash:016x}",
    }
    return result, np.asarray(thumbnail, dtype=np.uint8).copy()


def exact_duplicate_groups(records, hash_key):
    buckets = defaultdict(list)
    for record in records:
        if record.get(hash_key):
            buckets[record[hash_key]].append(record)
    return [{
        "paths": [record["path"] for record in group],
        "splits": sorted({record["split"] for record in group}),
        "cross_split": len({record["split"] for record in group}) > 1,
    } for group in buckets.values() if len(group) > 1]


def near_duplicate_candidates(records, thumbnails, max_images=3000):
    # Ce filtre est un outil de revue, pas une preuve de duplication ou d'absence de fuite.
    selected = list(records)
    sampled = len(selected) > max_images
    if sampled:
        selected = random.Random(SEED).sample(selected, max_images)
    candidates = []
    for index, first in enumerate(selected):
        for second in selected[index + 1:]:
            if first["pixel_sha256"] == second["pixel_sha256"]:
                continue
            distance = (int(first["phash63"], 16) ^ int(second["phash63"], 16)).bit_count()
            if distance > 6:
                continue
            a = thumbnails[first["path"]].astype(np.float32)
            b = thumbnails[second["path"]].astype(np.float32)
            difference = float(np.abs(a - b).mean() / 255)
            if difference <= 0.06:
                candidates.append({
                    "first": first["path"], "second": second["path"],
                    "hash_distance": distance, "thumbnail_mae": difference,
                    "cross_split": first["split"] != second["split"],
                })
    return sorted(candidates, key=lambda row: (not row["cross_split"], row["thumbnail_mae"])), sampled


def contact_sheet(root, records, destination, columns=5):
    if not records:
        return
    cell_width, cell_height = 280, 225
    rows = (len(records) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#ededed")
    draw = ImageDraw.Draw(sheet)
    for index, record in enumerate(records):
        rgb, _ = decode_image((Path(root) / record["path"]).read_bytes())
        rgb.thumbnail((cell_width - 12, cell_height - 45), Image.Resampling.LANCZOS)
        x, y = (index % columns) * cell_width, (index // columns) * cell_height
        sheet.paste(rgb, (x + (cell_width - rgb.width) // 2, y + 4))
        label = f"{Path(record['path']).name} | {record['split']} | L={record['mean_luminance']:.2f}"
        draw.text((x + 6, y + cell_height - 37), label, fill="black")
        draw.text((x + 6, y + cell_height - 21), f"entropy={record['entropy_bits']:.2f} | edge={record['laplacian_variance_256']:.4f}", fill="black")
    sheet.save(destination, quality=90)


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(root, output, min_size=256, progress=None):
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset absent : {root}")
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Le dossier de rapport doit être séparé du dataset source.")
    files = sorted(file for file in root.rglob("*") if file.is_file() and file.suffix.lower() in EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"Aucune image JPEG/PNG dans {root} ; vérifier le chemin Kaggle.")
    output.mkdir(parents=True, exist_ok=False)
    records, errors, thumbnails, fingerprints = [], [], {}, []
    iterator = progress(files, desc="Audit complet des images") if progress else files
    for file in iterator:
        relative = file.relative_to(root).as_posix()
        record = {"path": relative, "split": identify_split(relative), "image_id": file.stem}
        try:
            raw = file.read_bytes()
            checksum = hashlib.sha256(raw).hexdigest()
            fingerprints.append(f"{relative},{checksum}")
            rgb, info = decode_image(raw)
            features, thumbnail = image_features(rgb)
            record.update({"bytes": len(raw), "file_sha256": checksum, **info, **features})
            record["too_small"] = min(rgb.size) < min_size
            records.append(record)
            thumbnails[relative] = thumbnail
        except Exception as exc:
            errors.append({"path": relative, "error": str(exc)})
    file_duplicates = exact_duplicate_groups(records, "file_sha256")
    pixel_duplicates = exact_duplicate_groups(records, "pixel_sha256")
    near_duplicates, sampled = near_duplicate_candidates(records, thumbnails)
    missing_ids = sorted({f"{i:04d}" for i in range(1, 901)} - {row["image_id"] for row in records})
    id_counts = Counter(record["image_id"] for record in records)
    repeated_ids = sorted(name for name, count in id_counts.items() if count > 1)
    split_counts = dict(Counter(record["split"] for record in records))
    expected_counts = {"train": 720, "val": 80, "test": 100}
    compatible = (len(records) == 900 and not errors and not missing_ids and not repeated_ids
                  and not pixel_duplicates and not any(record["too_small"] for record in records)
                  and split_counts == expected_counts)
    fingerprint = hashlib.sha256("\n".join(fingerprints).encode()).hexdigest()
    summary = {
        "audit_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(), "source": str(root),
        "split_seed": SEED, "minimum_side_pixels": min_size,
        "files_found": len(files), "decoded_images": len(records), "errors_count": len(errors),
        "split_counts": split_counts, "source_modes": dict(Counter(row["source_mode"] for row in records)),
        "dimension_range": {
            "width": [min(row["width"] for row in records), max(row["width"] for row in records)],
            "height": [min(row["height"] for row in records), max(row["height"] for row in records)],
        } if records else None,
        "too_small_count": sum(row["too_small"] for row in records),
        "exact_file_duplicate_groups": len(file_duplicates), "exact_pixel_duplicate_groups": len(pixel_duplicates),
        "cross_split_exact_groups": sum(group["cross_split"] for group in pixel_duplicates),
        "near_duplicate_candidates": len(near_duplicates),
        "cross_split_near_candidates": sum(row["cross_split"] for row in near_duplicates),
        "near_duplicate_search_sampled": sampled,
        "near_duplicate_thresholds": {"phash63_hamming_max": 6, "thumbnail_mae_max": 0.06, "maximum_images": 3000},
        "missing_div2k_ids": missing_ids, "repeated_div2k_ids": repeated_ids,
        "compatible_with_current_900_image_loader": compatible,
        "dataset_fingerprint_sha256": fingerprint,
        "content_quality_status": "manual_review_required",
        "limitations": [
            "Perceptual similarity flags are candidates, not confirmed duplicates.",
            "No semantic scene labels or camera-noise fidelity are inferred.",
            "Brightness, entropy and sharpness proxies do not prove that a photo is bad.",
            "This audit cannot establish whether adding images improves model quality.",
        ],
    }
    write_csv(output / "images.csv", records)
    write_csv(output / "errors.csv", errors)
    write_csv(output / "near_duplicates.csv", near_duplicates)
    (output / "exact_duplicates.json").write_text(json.dumps({"file": file_duplicates, "pixels": pixel_duplicates}, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    sample = random.Random(SEED).sample(records, min(30, len(records)))
    contact_sheet(root, sample, output / "sample.jpg")
    low_edges = sorted(records, key=lambda row: row["laplacian_variance_256"])[:10]
    extremes = sorted(records, key=lambda row: max(row["dark_fraction"], row["bright_fraction"]), reverse=True)[:10]
    review = list({row["path"]: row for row in low_edges + extremes}.values())
    contact_sheet(root, review, output / "review_candidates.jpg")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Nouveau dossier, distinct des images.")
    args = parser.parse_args()
    summary = audit_dataset(args.data_root, args.output)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
