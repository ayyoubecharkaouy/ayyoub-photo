"""Tests hors ligne des notebooks, sans entraîner ni importer PyTorch."""

import ast
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
from types import SimpleNamespace
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS = ROOT / "ml" / "notebooks"
TRAIN = NOTEBOOKS / "01_ayyphoto_denoising_kaggle.ipynb"
INFER = NOTEBOOKS / "02_ayyphoto_inference_kaggle.ipynb"


def notebook_module(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return ast.parse("\n".join("".join(cell["source"]) for cell in notebook["cells"]
                               if cell["cell_type"] == "code"))


def definition(path, name):
    return next(node for node in notebook_module(path).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name)


def isolated(path, names, namespace):
    module = ast.Module(body=[definition(path, name) for name in names], type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def literal(path, name):
    return next(ast.literal_eval(node.value) for node in notebook_module(path).body
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))


class NotebookContractTests(unittest.TestCase):
    def test_notebooks_are_valid_and_not_executed(self):
        for path in NOTEBOOKS.glob("*.ipynb"):
            with self.subTest(path=path.name):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["nbformat"], 4)
                ids = [cell["id"] for cell in notebook["cells"]]
                self.assertEqual(len(ids), len(set(ids)))
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        ast.parse("".join(cell["source"]))
                        self.assertIsNone(cell["execution_count"])
                        self.assertEqual(cell["outputs"], [])

    def test_inference_keeps_the_trained_architecture(self):
        self.assertEqual(
            ast.dump(definition(TRAIN, "AyyPhotoDenoiser")),
            ast.dump(definition(INFER, "AyyPhotoDenoiser")),
        )
        self.assertEqual(ast.dump(definition(TRAIN, "load_rgb")), ast.dump(definition(INFER, "load_rgb")))
        calls = [node.func for node in ast.walk(notebook_module(INFER)) if isinstance(node, ast.Call)]
        self.assertFalse(any(isinstance(call, ast.Attribute) and call.attr in
                             {"backward", "step", "mount", "dataset_download"} for call in calls))

    def test_final_test_is_opt_in_and_fingerprinted(self):
        self.assertFalse(literal(TRAIN, "RUN_FINAL_TEST"))
        source = TRAIN.read_text(encoding="utf-8")
        self.assertIn("test_protocol.json", source)
        self.assertIn("weights_sha256", source)
        self.assertIn("manifest_sha256", source)


class ArtifactTests(unittest.TestCase):
    def namespace(self):
        class JsonTorch:
            # Doubles d'IO : les tests ne prétendent pas valider la sérialisation tensorielle.
            @staticmethod
            def load(path, **kwargs):
                if kwargs != {"map_location": "cpu", "weights_only": True}:
                    raise AssertionError(kwargs)
                return json.loads(Path(path).read_text())

            @staticmethod
            def save(payload, path):
                Path(path).write_text(json.dumps(payload))

        namespace = {
            "Path": Path, "datetime": datetime, "timezone": timezone,
            "shutil": shutil, "zipfile": zipfile, "json": json, "hashlib": hashlib,
            "torch": JsonTorch, "CFG": literal(TRAIN, "CFG"),
            "RUN_ARTIFACTS": literal(TRAIN, "RUN_ARTIFACTS"),
        }
        return isolated(TRAIN, ["file_sha256", "atomic_torch_save", "writable_run_copy", "export_run_archive"], namespace)

    def test_readonly_input_is_copied_to_a_distinct_writable_run(self):
        ns = self.namespace()
        with tempfile.TemporaryDirectory(prefix="ayyphoto-resume-") as temporary:
            root = Path(temporary)
            source, outputs = root / "input" / "old_run", root / "working" / "experiments"
            source.mkdir(parents=True)
            for name in ("config.json", "best_weights.pth", "last.ckpt"):
                (source / name).write_text("original-" + name)
            (source / "private_photo.png").write_text("must not copy")
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            restored = ns["writable_run_copy"](source, outputs)
            self.assertIn(outputs.resolve(), restored.parents)
            self.assertEqual({p.name for p in restored.iterdir()}, {"config.json", "best_weights.pth", "last.ckpt"})
            self.assertEqual(before, {p.name: p.read_bytes() for p in source.iterdir()})
            self.assertEqual(ns["writable_run_copy"](restored, outputs), restored)

    def test_archive_has_portable_bundle_and_no_photos(self):
        ns = self.namespace()
        with tempfile.TemporaryDirectory(prefix="ayyphoto-export-") as temporary:
            run = Path(temporary) / "run"
            run.mkdir()
            (run / "best_weights.pth").write_text(json.dumps({"network.0.weight": "tensor-double"}))
            (run / "config.json").write_text("{}")
            (run / "last.ckpt").write_text("resume-double")
            (run / "private_photo.png").write_bytes(b"not included")
            archive = ns["export_run_archive"](run)
            with zipfile.ZipFile(archive) as exported:
                self.assertEqual(set(exported.namelist()), {
                    "run/best_weights.pth", "run/model_bundle.pt", "run/config.json", "run/last.ckpt",
                })
                bundle = json.loads(exported.read("run/model_bundle.pt"))
                self.assertEqual(bundle["config"]["depth"], literal(TRAIN, "CFG")["depth"])
                self.assertEqual(bundle["architecture"], "residual_rgb_cnn")
            (run / "status.json").write_text('{"state":"ready"}')
            self.assertEqual(ns["export_run_archive"](run), archive)
            with zipfile.ZipFile(archive) as exported:
                self.assertIn("run/status.json", exported.namelist())
            self.assertFalse(archive.with_name(archive.name + ".tmp").exists())

    def test_rng_restore_tolerates_fewer_available_gpus(self):
        restored = []
        ns = {
            "DEVICE": SimpleNamespace(type="cuda"),
            "random": SimpleNamespace(setstate=lambda state: None),
            "np": SimpleNamespace(array=lambda state, dtype: state, uint32=int,
                                  random=SimpleNamespace(set_state=lambda state: None)),
            "loader_generator": SimpleNamespace(set_state=lambda state: None),
            "torch": SimpleNamespace(set_rng_state=lambda state: None,
                cuda=SimpleNamespace(device_count=lambda: 1,
                    set_rng_state=lambda state, device: restored.append((device, state)))),
        }
        restore = isolated(TRAIN, ["restore_rng"], ns)["restore_rng"]
        restore({"python": None, "numpy": ["MT19937", [1], 0, 0, 0],
                 "torch": "cpu-state", "loader": "loader-state", "cuda": ["gpu-0", "gpu-1"]})
        self.assertEqual(restored, [(0, "gpu-0")])


class InputAndModelTests(unittest.TestCase):
    def test_nested_input_and_ambiguity(self):
        find = isolated(TRAIN, ["find_div2k_input"], {"Path": Path})["find_div2k_input"]
        with tempfile.TemporaryDirectory(prefix="ayyphoto-input-") as temporary:
            root = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                find(root)
            dataset = root / "datasets" / "owner" / "div2k"
            for name in ("DIV2K_train_HR", "DIV2K_valid_HR"):
                (dataset / name / name).mkdir(parents=True)
            self.assertEqual(find(root), dataset.resolve())
            for name in ("DIV2K_train_HR", "DIV2K_valid_HR"):
                (root / "other_copy" / name).mkdir(parents=True)
            with self.assertRaises(ValueError):
                find(root)

    def test_inference_discovers_one_artifact_per_experiment(self):
        discover = isolated(INFER, ["discover_model_files"], {"Path": Path})["discover_model_files"]
        with tempfile.TemporaryDirectory(prefix="ayyphoto-weights-") as temporary:
            root = Path(temporary)
            (root / "best_weights.pth").touch()
            (root / "model_bundle.pt").touch()
            (root / "run_export.zip").touch()
            self.assertEqual(discover([root]), [(root / "model_bundle.pt").resolve()])
            other = root / "another_run"
            other.mkdir()
            (other / "best_weights.pth").touch()
            self.assertEqual(len(discover([root])), 2)

    def test_architecture_is_derived_from_actual_weight_shapes(self):
        infer = isolated(INFER, ["infer_architecture"], {"re": re})["infer_architecture"]
        channels, depth = 24, 5
        weights = {f"network.{2 * index}.weight": SimpleNamespace(shape=
            (channels, 3, 3, 3) if index == 0 else
            ((3, channels, 3, 3) if index == depth - 1 else (channels, channels, 3, 3)))
            for index in range(depth)}
        self.assertEqual(infer(weights), {"channels": channels, "depth": depth})
        weights["network.2.weight"] = SimpleNamespace(shape=(channels, channels, 5, 5))
        with self.assertRaises(ValueError):
            infer(weights)

    def test_zip_model_loading_does_not_extract_files(self):
        def load(stream, **kwargs):
            self.assertEqual(kwargs, {"map_location": "cpu", "weights_only": True})
            return json.loads(stream.read())

        ns = {"Path": Path, "zipfile": zipfile, "io": io, "hashlib": hashlib,
              "torch": SimpleNamespace(load=load)}
        read = isolated(INFER, ["read_model_payload"], ns)["read_model_payload"]
        with tempfile.TemporaryDirectory(prefix="ayyphoto-zip-model-") as temporary:
            root = Path(temporary)
            archive = root / "model_export.zip"
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("run/best_weights.pth", '{"which":"weights"}')
                output.writestr("run/model_bundle.pt", '{"which":"bundle"}')
                output.writestr("../unexpected.txt", "never extracted")
            before = set(root.iterdir())
            payload, origin, checksum = read(archive)
            self.assertEqual(payload, {"which": "bundle"})
            self.assertTrue(origin.endswith("!run/model_bundle.pt"))
            self.assertEqual(checksum, hashlib.sha256(b'{"which":"bundle"}').hexdigest())
            self.assertEqual(before, set(root.iterdir()))


class DatasetParityTests(unittest.TestCase):
    def test_crop_cache_preserves_training_and_validation_pairs(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy nécessaire pour vérifier les paires synthétiques")

        class Frame:
            def __init__(self, rows):
                self.rows = rows
                self.iloc = self

            def reset_index(self, drop):
                return self

            def __len__(self):
                return len(self.rows)

            def __getitem__(self, index):
                return self.rows[index]

        generator = np.random.default_rng(123)
        images = {f"{i}.png": generator.integers(0, 256, size=(384, 320, 3), dtype=np.uint8) for i in range(3)}
        original_hashes = {name: hashlib.sha256(image.tobytes()).hexdigest() for name, image in images.items()}
        frame = Frame([{"path": name, "scene": name} for name in images])
        reads = []

        def read(path):
            reads.append(path.name)
            return images[path.name]

        namespace = {"Dataset": object, "np": np, "torch": SimpleNamespace(from_numpy=lambda value: value),
                     "CFG": literal(TRAIN, "CFG"), "DATA_ROOT": Path("/unused"), "load_rgb": read,
                     "OrderedDict": OrderedDict, "RGB_CACHE_MIB": 0.5}
        new = isolated(TRAIN, ["SyntheticDenoisingDataset"], dict(namespace))["SyntheticDenoisingDataset"]
        old = isolated(NOTEBOOKS / "01_ayyphoto_denoising_drive.ipynb",
                       ["SyntheticDenoisingDataset"], dict(namespace))["SyntheticDenoisingDataset"]
        for training in (True, False):
            actual, expected = new(frame, training=training), old(frame, training=training)
            actual.epoch = expected.epoch = 3
            for index in range(len(actual)):
                a, b = actual[index], expected[index]
                np.testing.assert_array_equal(a["clean"], b["clean"])
                np.testing.assert_array_equal(a["noisy"], b["noisy"])
                self.assertEqual(a["sigma"], b["sigma"])
                self.assertLessEqual(actual._cache_bytes, actual._cache_limit)
        cached = new(frame, training=True)
        reads.clear()
        cached[0]
        cached[1]
        self.assertEqual(reads, ["0.png"])
        cached[8]
        self.assertEqual(len(cached._rgb_cache), 1)
        self.assertEqual(original_hashes, {name: hashlib.sha256(image.tobytes()).hexdigest() for name, image in images.items()})


if __name__ == "__main__":
    unittest.main()
