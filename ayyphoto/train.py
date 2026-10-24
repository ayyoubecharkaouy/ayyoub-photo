"""Entraînement local sur CPU, déclenché explicitement par CLI ou Streamlit."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import traceback
import zipfile

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainConfig, read_json, write_json
from .data import SyntheticPairs, check_dimensions, fingerprint, index_images, select_records
from .download import prepare_training_data
from .runtime import AyyPhotoDenoiser, load_model_path


def save_torch(value, destination):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(destination)


def archive_run(directory, config):
    directory = Path(directory)
    best = directory / "best_weights.pth"
    if best.is_file():
        save_torch({
            "format_version": 1, "architecture": "residual_rgb_cnn",
            "model_state_dict": torch.load(best, map_location="cpu", weights_only=True),
            "config": {"channels": config.channels, "depth": config.depth},
            "preprocessing": {"color_space": "sRGB", "input_range": [0, 1], "output_clamp": [0, 1]},
            "model_version": "0.2.0-local-candidate",
        }, directory / "model_bundle.pt")
    destination = directory / "model_export.zip"
    temporary = directory / "model_export.zip.tmp"
    names = ("best_weights.pth", "model_bundle.pt", "last.ckpt", "interrupted_weights.pth",
             "config.json", "history.csv", "summary.json", "status.json", "manifest.json")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for name in names:
            path = directory / name
            if path.is_file():
                output.write(path, arcname=f"{directory.name}/{name}")
    temporary.replace(destination)
    return destination


def save_history(directory, history):
    if not history:
        return
    temporary = Path(directory) / "history.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(Path(directory) / "history.csv")


@torch.inference_mode()
def evaluate(model, dataset):
    model.eval()
    values = []
    for noisy, clean in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                   generator=torch.Generator().manual_seed(0)):
        raw = model(noisy).float()
        prediction = raw.clamp(0, 1)
        a, b = clean[0].permute(1, 2, 0).numpy(), prediction[0].permute(1, 2, 0).numpy()
        values.append({
            "l1": float(nn.functional.l1_loss(raw, clean)),
            "psnr": float(peak_signal_noise_ratio(a, b, data_range=1.0)),
            "ssim": float(structural_similarity(a, b, data_range=1.0, channel_axis=-1)),
        })
    return {key: float(np.mean([row[key] for row in values])) for key in values[0]}


def train(config, directory):
    config.validate()
    directory = Path(directory)
    prepare_training_data(config, directory)
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    records = index_images(config.data_root, config.seed)
    data_hash = fingerprint(config.data_root, records)
    train_rows = select_records(records, "train", config.max_train_images, config.seed)
    val_rows = select_records(records, "val", config.max_val_images, config.seed)
    check_dimensions(config.data_root, train_rows, config.patch_size)
    check_dimensions(config.data_root, val_rows, config.eval_size)
    training = SyntheticPairs(config.data_root, train_rows, config, training=True)
    validation = SyntheticPairs(config.data_root, val_rows, config)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(training, batch_size=config.batch_size, shuffle=True, num_workers=0, generator=generator)
    model = AyyPhotoDenoiser(config.channels, config.depth)
    if config.initial_model:
        loaded = load_model_path(config.initial_model, config.threads)
        if loaded.architecture != {"channels": config.channels, "depth": config.depth}:
            raise ValueError("Les poids initiaux ont une architecture différente de la configuration.")
        model.load_state_dict(loaded.model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    start, best, bad_epochs, history = 0, -float("inf"), 0, []
    if config.resume_run:
        source = Path(config.resume_run).expanduser().resolve()
        checkpoint = torch.load(source / "last.ckpt", map_location="cpu", weights_only=True)
        if checkpoint.get("pipeline") != "ayyphoto.local.v1":
            raise ValueError("Pour des poids Kaggle, utiliser les poids initiaux. La reprise complète attend une expérience locale.")
        if checkpoint["data_fingerprint"] != data_hash:
            raise ValueError("Le dataset a changé depuis le checkpoint ; créer une nouvelle expérience.")
        ignored = {"epochs", "threads", "rgb_cache_mib", "data_root", "resume_run", "initial_model", "prepare_div2k"}
        differences = [key for key, value in config.to_dict().items()
                       if key not in ignored and json.dumps(value) != json.dumps(checkpoint["config"].get(key))]
        if differences:
            raise ValueError(f"Configuration différente de la reprise : {differences}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start, best, bad_epochs = checkpoint["epoch"] + 1, checkpoint["best_val_psnr"], checkpoint["bad_epochs"]
        history = checkpoint["history"]
        random.setstate(checkpoint["rng_python"])
        state = checkpoint["rng_numpy"]
        np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        torch.set_rng_state(checkpoint["rng_torch"])
        generator.set_state(checkpoint["rng_loader"])
        previous_best = source / "best_weights.pth"
        if not previous_best.is_file():
            raise ValueError("Le dossier de reprise doit aussi contenir best_weights.pth.")
        (directory / "best_weights.pth").write_bytes(previous_best.read_bytes())
        (directory / "last.ckpt").write_bytes((source / "last.ckpt").read_bytes())
    write_json(directory / "config.json", {"config": config.to_dict(), "data_fingerprint": data_hash,
                                           "architecture": "residual_rgb_cnn", "torch": str(torch.__version__)})
    write_json(directory / "manifest.json", {"records": [record.__dict__ for record in records],
                                             "train_selected": len(train_rows), "val_selected": len(val_rows)})
    baseline = []
    for noisy, clean in validation:
        mse = float(torch.mean((noisy-clean)**2))
        baseline.append(-10*np.log10(max(mse, 1e-12)))
    baseline_psnr = float(np.mean(baseline))
    print(f"CPU : {config.threads} threads ; train={len(train_rows)}, val={len(val_rows)}, référence={baseline_psnr:.3f} dB", flush=True)
    total_epochs = start if bad_epochs >= config.patience else config.epochs
    try:
        for epoch in range(start, total_epochs):
            started = time.monotonic()
            training.epoch = epoch
            model.train()
            loss_sum, samples = 0.0, 0
            for step, (noisy, clean) in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                prediction = model(noisy)
                loss = nn.functional.l1_loss(prediction, clean)
                if not torch.isfinite(loss):
                    raise FloatingPointError("La perte n’est plus finie ; vérifier les données et les réglages.")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                loss_sum += float(loss.detach()) * noisy.shape[0]
                samples += noisy.shape[0]
                if step % 10 == 0 or step == len(loader)-1:
                    write_json(directory / "status.json", {
                        "state": "training", "epoch": epoch+1, "total_epochs": config.epochs,
                        "step": step+1, "total_steps": len(loader), "train_l1": loss_sum/samples,
                        "message": f"Époque {epoch+1}/{config.epochs} — lot {step+1}/{len(loader)}",
                    })
            scores = evaluate(model, validation)
            if not all(np.isfinite(value) for value in scores.values()):
                raise FloatingPointError("Les métriques de validation ne sont pas finies.")
            scheduler.step(scores["psnr"])
            if scores["psnr"] > best:
                best, bad_epochs = scores["psnr"], 0
                save_torch(model.state_dict(), directory / "best_weights.pth")
            else:
                bad_epochs += 1
            history.append({"epoch": epoch+1, "train_l1": loss_sum/samples,
                            "val_l1": scores["l1"], "val_psnr": scores["psnr"], "val_ssim": scores["ssim"],
                            "val_gain_vs_noisy_db": scores["psnr"]-baseline_psnr,
                            "seconds": time.monotonic()-started})
            state = np.random.get_state()
            save_torch({
                "pipeline": "ayyphoto.local.v1", "epoch": epoch,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "best_val_psnr": best,
                "bad_epochs": bad_epochs, "history": history, "config": config.to_dict(),
                "data_fingerprint": data_hash, "rng_python": random.getstate(),
                "rng_numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
                "rng_torch": torch.get_rng_state(), "rng_loader": generator.get_state(),
            }, directory / "last.ckpt")
            save_history(directory, history)
            archive_run(directory, config)
            print(f"Époque {epoch+1} : L1={loss_sum/samples:.5f}, PSNR={scores['psnr']:.3f}, SSIM={scores['ssim']:.4f}", flush=True)
            if bad_epochs >= config.patience:
                print("Arrêt anticipé.", flush=True)
                break
    except KeyboardInterrupt:
        save_torch(model.state_dict(), directory / "interrupted_weights.pth")
        write_json(directory / "status.json", {"state": "cancelled", "message": "Arrêt demandé. Reprise possible à la dernière époque complète si last.ckpt existe."})
        archive_run(directory, config)
        return
    if not history:
        raise ValueError("Aucune époque terminée ; vérifier le nombre d’époques et le checkpoint.")
    save_history(directory, history)
    write_json(directory / "summary.json", {
        "pipeline": "ayyphoto.local.v1", "candidate_only": True, "config": config.to_dict(),
        "train_images": len(train_rows), "val_images": len(val_rows), "epochs_completed": len(history),
        "baseline_noisy_psnr": baseline_psnr, "best_validation_psnr": best,
        "best_gain_vs_noisy_db": best-baseline_psnr, "data_fingerprint": data_hash,
        "protocol": "synthetic_gaussian_center_crops_no_final_test", "device": "cpu",
    })
    write_json(directory / "status.json", {"state": "completed", "message": "Entraînement terminé. Les poids et rapports sont sauvegardés.",
                                           "epoch": len(history), "total_epochs": config.epochs, "best_val_psnr": best})
    archive_run(directory, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    directory = args.config.resolve().parent
    try:
        payload = json.loads(args.config.read_text(encoding="utf-8"))
        train(TrainConfig.from_dict(payload), directory)
        return 0
    except KeyboardInterrupt:
        write_json(directory / "status.json", {"state": "cancelled", "message": "Calcul arrêté pendant la préparation."})
        return 130
    except Exception as exc:
        write_json(directory / "status.json", {"state": "error", "message": str(exc)})
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
