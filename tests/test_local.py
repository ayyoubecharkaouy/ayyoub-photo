"""Contrôles locaux ; les tests PyTorch/Streamlit s'activent après installation."""

import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from ayyphoto.config import TrainConfig, read_json, write_json
from ayyphoto import jobs


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None
HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None
ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_local_model_matches_existing_kaggle_architecture(self):
        runtime = ast.parse((ROOT/"ayyphoto/runtime.py").read_text(encoding="utf-8"))
        notebook = json.loads((ROOT/"ml/notebooks/01_ayyphoto_denoising_kaggle.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        local = next(node for node in runtime.body if isinstance(node, ast.ClassDef) and node.name == "AyyPhotoDenoiser")
        previous = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "AyyPhotoDenoiser")
        self.assertEqual(ast.dump(local), ast.dump(previous))

    def test_cpu_defaults_and_invalid_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TrainConfig(data_root=directory).validate()
            self.assertEqual(config.batch_size, 1)
            self.assertEqual(config.threads, 2)
            self.assertEqual(config.patch_size, 64)
            for options in ({"epochs": 0}, {"patch_size": 8}, {"eval_size": 32}, {"sigma_max": -1}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    TrainConfig(data_root=directory, **options).validate()
            with self.assertRaises(ValueError):
                TrainConfig.from_dict({"data_root": directory, "unknown": True})

    def test_automatic_div2k_accepts_missing_destination_but_personal_data_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"missing"
            self.assertTrue(TrainConfig(data_root=str(target), prepare_div2k=True).validate().prepare_div2k)
            self.assertFalse(target.exists())
            with self.assertRaises(ValueError):
                TrainConfig(data_root=str(target)).validate()
            target.write_text("not a directory")
            with self.assertRaises(ValueError):
                TrainConfig(data_root=str(target), prepare_div2k=True).validate()

    def test_json_roundtrip_does_not_replace_valid_file_on_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"status.json"
            write_json(path, {"state": "running"})
            with self.assertRaises(ValueError):
                write_json(path, {"loss": float("nan")})
            self.assertEqual(read_json(path), {"state": "running"})
            self.assertEqual(read_json(Path(directory)/"missing", {}), {})


class JobTests(unittest.TestCase):
    def test_unrelated_process_is_not_cancelled(self):
        job = {"pid": os.getpid(), "kind": "train", "config_path": "/nonexistent/ayyphoto/job_config.json"}
        self.assertFalse(jobs.is_running(job))
        with mock.patch.object(jobs.os, "kill") as kill:
            self.assertFalse(jobs.stop_job(job))
            kill.assert_not_called()

    def test_child_uses_current_python_and_json_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)/"experiments"
            config = {"data_root": "/tmp/Photos de vacances", "threads": 2}
            with mock.patch.object(jobs, "EXPERIMENTS", outputs), mock.patch.object(jobs, "ACTIVE_JOB", outputs/"active.json"), \
                 mock.patch.object(jobs.subprocess, "Popen", return_value=mock.Mock(pid=4242)) as popen:
                job = jobs.start_job("train", config)
                arguments, kwargs = popen.call_args
                self.assertEqual(arguments[0][:3], [jobs.sys.executable, "-m", "ayyphoto.train"])
                self.assertFalse(kwargs.get("shell", False))
                self.assertTrue(kwargs["start_new_session"])
                self.assertEqual(read_json(job["config_path"]), config)
                self.assertEqual(read_json(outputs/"active.json")["pid"], 4242)
                self.assertTrue((Path(job["run_dir"])/"run.log").is_file())

    def test_second_running_job_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            outputs = Path(directory)
            with mock.patch.object(jobs, "EXPERIMENTS", outputs), mock.patch.object(jobs, "ACTIVE_JOB", outputs/"active.json"), \
                 mock.patch.object(jobs, "is_running", return_value=True), mock.patch.object(jobs.subprocess, "Popen") as popen:
                with self.assertRaises(ValueError):
                    jobs.start_job("train", {})
                popen.assert_not_called()


@unittest.skipUnless(HAS_NUMPY and HAS_PIL, "NumPy et Pillow requis")
class ImagingTests(unittest.TestCase):
    def test_photo_roundtrip_and_original_preserved(self):
        import numpy as np
        from PIL import Image
        from ayyphoto.imaging import decode_photo, png_bytes
        rgb = np.random.default_rng(42).integers(0, 256, (24, 31, 3), dtype=np.uint8)
        raw = png_bytes(rgb)
        np.testing.assert_array_equal(decode_photo(raw), rgb)
        with Image.open(io.BytesIO(raw)) as image:
            self.assertTrue(image.info.get("icc_profile"))
            self.assertFalse(image.getexif())

    def test_corrupt_transparent_and_oversized_images(self):
        from PIL import Image
        from ayyphoto import imaging
        with self.assertRaises(ValueError):
            imaging.decode_photo(b"not a photo")
        output = io.BytesIO()
        Image.new("RGBA", (10, 10)).save(output, format="PNG")
        with self.assertRaises(ValueError):
            imaging.decode_photo(output.getvalue())
        output = io.BytesIO()
        Image.new("RGB", (10, 10)).save(output, format="PNG")
        with mock.patch.object(imaging, "MAX_PHOTO_PIXELS", 99), self.assertRaises(ValueError):
            imaging.decode_photo(output.getvalue())

    def test_exif_orientation_is_applied(self):
        from PIL import Image
        from ayyphoto.imaging import decode_photo
        image = Image.new("RGB", (7, 3), "red")
        exif = Image.Exif()
        exif[274] = 6
        stream = io.BytesIO()
        image.save(stream, format="JPEG", exif=exif)
        self.assertEqual(decode_photo(stream.getvalue()).shape, (7, 3, 3))


@unittest.skipUnless(HAS_TORCH and HAS_NUMPY and HAS_PIL, "Installer les dépendances CPU dans .venv")
class RuntimeTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_old_weights_bundle_checkpoint_and_zip_load(self):
        import torch
        from ayyphoto.runtime import AyyPhotoDenoiser, load_model
        model = AyyPhotoDenoiser(channels=4, depth=3)
        for value in (model.state_dict(), {"model_state_dict": model.state_dict(), "config": {"channels": 4, "depth": 3}}):
            stream = io.BytesIO()
            torch.save(value, stream)
            loaded = load_model(stream.getvalue())
            self.assertEqual(loaded.architecture, {"channels": 4, "depth": 3})
            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("run/best_weights.pth", stream.getvalue())
                output.writestr("../ignored.txt", "not extracted")
            self.assertEqual(load_model(archive.getvalue()).architecture, loaded.architecture)

    def test_tiled_prediction_matches_full_image_and_strength_zero(self):
        import numpy as np
        import torch
        from ayyphoto.runtime import AyyPhotoDenoiser, denoise, load_model
        model = AyyPhotoDenoiser(channels=4, depth=3)
        with torch.no_grad():
            model.network[-1].weight.normal_(0, 0.02)
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        loaded = load_model(stream.getvalue(), threads=1)
        rgb = np.random.default_rng(4).integers(0, 256, (73, 87, 3), dtype=np.uint8)
        with torch.inference_mode():
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0)/255
            full = loaded.model(tensor)[0].clamp(0, 1).permute(1, 2, 0).numpy()
        expected = np.rint(full*255).astype(np.uint8)
        actual = denoise(loaded, rgb, tile_size=16)
        self.assertLessEqual(int(np.abs(actual.astype(int)-expected.astype(int)).max()), 1)
        np.testing.assert_array_equal(denoise(loaded, rgb, strength=0), rgb)

    def test_invalid_or_nonfinite_weights_are_rejected(self):
        import torch
        from ayyphoto.runtime import AyyPhotoDenoiser, load_model
        with self.assertRaises(ValueError):
            load_model(b"invalid")
        model = AyyPhotoDenoiser(channels=4, depth=2)
        with torch.no_grad():
            model.network[-1].weight.fill_(float("nan"))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        with self.assertRaises(ValueError):
            load_model(stream.getvalue())


@unittest.skipUnless(HAS_TORCH and HAS_SKIMAGE and HAS_PIL, "PyTorch et scikit-image requis")
class TrainingSmokeTests(unittest.TestCase):
    def test_one_epoch_resume_and_portable_export(self):
        import numpy as np
        import torch
        from PIL import Image
        from ayyphoto.train import train
        from ayyphoto.runtime import load_model_path
        with tempfile.TemporaryDirectory(prefix="ayyphoto-train-smoke-") as temporary:
            root = Path(temporary)
            data = root/"data"
            for index, split in enumerate(("train", "val")):
                directory = data/split/f"scene_{split}"
                directory.mkdir(parents=True)
                rgb = np.random.default_rng(index).integers(0, 256, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(directory/"photo.png")
            first, second = root/"first", root/"second"
            first.mkdir()
            second.mkdir()
            config = TrainConfig(data_root=str(data), epochs=1, patch_size=16, eval_size=16,
                                 patches_per_image=1, channels=4, depth=2, threads=1)
            train(config, first)
            self.assertEqual(read_json(first/"status.json")["state"], "completed")
            self.assertTrue((first/"model_export.zip").is_file())
            self.assertEqual(load_model_path(first/"model_bundle.pt", threads=1).architecture, {"channels": 4, "depth": 2})
            config.epochs, config.resume_run = 2, str(first)
            train(config, second)
            checkpoint = torch.load(second/"last.ckpt", map_location="cpu", weights_only=True)
            self.assertEqual(len(checkpoint["history"]), 2)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertTrue((first/"best_weights.pth").is_file())


@unittest.skipUnless(HAS_STREAMLIT and HAS_TORCH and HAS_SKIMAGE, "Installer Streamlit et PyTorch dans .venv")
class InterfaceTests(unittest.TestCase):
    def test_training_with_missing_data_starts_automatic_preparation(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch("ayyphoto.jobs.start_job", return_value={"run_dir": str(root/"run")}) as start, \
                 mock.patch("ayyphoto.jobs.active_job", return_value=None):
                app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
                next(field for field in app.text_input if field.label == "Dossier local des images").set_value(str(root/"missing"))
                next(button for button in app.button if button.label == "Lancer l’entraînement CPU").click().run()
                self.assertEqual(len(app.exception), 0)
                start.assert_called_once()
                kind, config = start.call_args.args
                self.assertEqual(kind, "train")
                self.assertTrue(config["prepare_div2k"])
                self.assertEqual(config["data_root"], str(root/"missing"))
                self.assertFalse((root/"missing").exists())

    def test_empty_local_interface_does_not_start_training(self):
        from streamlit.testing.v1 import AppTest
        with mock.patch("ayyphoto.jobs.subprocess.Popen") as popen:
            app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            buttons = [button for button in app.button if button.label == "Corriger la photo"]
            self.assertEqual(len(buttons), 1)
            self.assertTrue(buttons[0].disabled)
            popen.assert_not_called()

    def test_saved_experiment_monitor_renders(self):
        from streamlit.testing.v1 import AppTest
        from ayyphoto import config
        with tempfile.TemporaryDirectory() as temporary:
            outputs = Path(temporary)
            run = outputs/"train_test"
            run.mkdir()
            write_json(run/"status.json", {"state": "completed", "message": "Test terminé"})
            (run/"history.csv").write_text("epoch,train_l1,val_l1,val_psnr,val_ssim\n1,0.1,0.1,30.0,0.9\n")
            (run/"model_bundle.pt").write_bytes(b"download fixture")
            with mock.patch.object(config, "EXPERIMENTS", outputs), mock.patch.object(jobs, "active_job", return_value=None):
                app = AppTest.from_file(str(ROOT/"app.py"), default_timeout=30).run()
                self.assertEqual(len(app.exception), 0)
                self.assertTrue(any(metric.label == "Meilleur PSNR validation" for metric in app.metric))


if __name__ == "__main__":
    unittest.main()
