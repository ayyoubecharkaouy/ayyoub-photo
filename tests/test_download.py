"""Préparation de DIV2K testée sans réseau avec de petits fichiers temporaires."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

try:
    from PIL import Image
except ImportError:
    Image = None

from ayyphoto.download import inspect_dataset, inventory, prepare_dataset
from ayyphoto.download import prepare_training_data
from ayyphoto.config import TrainConfig, read_json


ROOT = Path(__file__).resolve().parents[1]
SMALL_SPEC = {"0001": "DIV2K_train_HR", "0801": "DIV2K_valid_HR"}


def make_dataset(root, expected):
    paths = {}
    for index, (image_id, group) in enumerate(expected.items()):
        path = Path(root)/group/group/f"{image_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (256, 256), (index*67 % 255, index*91 % 255, 127)).save(path)
        paths[image_id] = path
    return paths


class AutomaticPreparationTests(unittest.TestCase):
    def test_personal_dataset_never_triggers_download(self):
        with mock.patch("ayyphoto.download.prepare_dataset") as prepare:
            prepare_training_data(TrainConfig(data_root="/unused"), "/unused")
            prepare.assert_not_called()

    def test_automatic_preparation_reports_readiness_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = TrainConfig(data_root=str(root/"missing"), prepare_div2k=True).validate()
            report = {"complete": True, "valid_images": 900, "installed_files": 900}

            def prepare(destination, progress):
                self.assertEqual(read_json(root/"run/status.json")["state"], "verifying")
                progress({"state": "downloading", "message": "Téléchargement"})
                return report

            with mock.patch("ayyphoto.download.prepare_dataset", side_effect=prepare) as mocked:
                prepare_training_data(config, root/"run")
            mocked.assert_called_once()
            self.assertEqual(read_json(root/"run/dataset_report.json"), report)
            self.assertEqual(read_json(root/"run/status.json")["state"], "starting")

    def test_preparation_failure_prevents_ready_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = TrainConfig(data_root=str(root/"missing"), prepare_div2k=True)
            with mock.patch("ayyphoto.download.prepare_dataset", side_effect=RuntimeError("offline")):
                with self.assertRaises(RuntimeError):
                    prepare_training_data(config, root/"run")
            self.assertFalse((root/"run/dataset_report.json").exists())
            self.assertNotEqual(read_json(root/"run/status.json")["state"], "starting")


class DownloadCommandTests(unittest.TestCase):
    def test_check_of_missing_data_never_downloads_or_creates_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/"missing data"
            result = subprocess.run([sys.executable, str(ROOT/"scripts/download_data.py"),
                                     "--destination", str(target), "--check"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["expected_images"], 900)
            self.assertEqual(report["missing_images"], 900)
            self.assertFalse(target.exists())


@unittest.skipUnless(Image is not None, "Pillow requis pour vérifier les images")
class DownloadTests(unittest.TestCase):
    def test_valid_existing_dataset_does_not_contact_kaggle(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/"data"
            files = make_dataset(destination, SMALL_SPEC)
            hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()}
            downloader = mock.Mock(side_effect=AssertionError("No network expected"))
            result = prepare_dataset(destination, downloader=downloader, expected=SMALL_SPEC)
            self.assertTrue(result["complete"])
            self.assertFalse(result["downloaded"])
            self.assertEqual(result["installed_files"], 0)
            downloader.assert_not_called()
            self.assertEqual(hashes, {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in files.items()})

    def test_partial_dataset_downloads_only_missing_image_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_dataset(root/"cache", SMALL_SPEC)
            destination = root/"data"
            existing = make_dataset(destination, {"0001": SMALL_SPEC["0001"]})["0001"]
            before = existing.read_bytes()
            downloader = mock.Mock(return_value=source["0801"])
            result = prepare_dataset(destination, downloader=downloader, expected=SMALL_SPEC)
            downloader.assert_called_once_with("DIV2K_valid_HR/DIV2K_valid_HR/0801.png")
            self.assertEqual(existing.read_bytes(), before)
            self.assertTrue(result["complete"])
            self.assertEqual(result["installed_files"], 1)
            downloader.reset_mock()
            again = prepare_dataset(destination, downloader=downloader, expected=SMALL_SPEC)
            self.assertFalse(again["downloaded"])
            downloader.assert_not_called()

    def test_corrupt_file_is_backed_up_before_replacement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = make_dataset(root/"cache", SMALL_SPEC)
            target = make_dataset(root/"data", SMALL_SPEC)
            target["0801"].write_bytes(b"corrupted photo")
            before = inspect_dataset(root/"data", SMALL_SPEC)
            self.assertEqual(len(before.damaged), 1)
            result = prepare_dataset(root/"data", downloader=mock.Mock(return_value=source["0801"]), expected=SMALL_SPEC)
            self.assertTrue(result["complete"])
            self.assertEqual(target["0801"].read_bytes(), source["0801"].read_bytes())
            backups = list(target["0801"].parent.glob("0801.png.invalid_*"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), b"corrupted photo")

    def test_bulk_download_is_used_for_initial_preparation(self):
        spec = {f"{index:04d}": "DIV2K_train_HR" for index in range(1, 35)}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root/"cache", spec)
            downloader = mock.Mock(return_value=root/"cache")
            result = prepare_dataset(root/"data", downloader=downloader, expected=spec)
            self.assertTrue(result["complete"])
            self.assertEqual(result["installed_files"], len(spec))
            downloader.assert_called_once_with()

    def test_changed_remote_layout_falls_back_to_full_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_dataset(root/"cache", SMALL_SPEC)
            make_dataset(root/"data", {"0001": SMALL_SPEC["0001"]})
            downloader = mock.Mock(side_effect=[FileNotFoundError("remote path changed"), root/"cache"])
            result = prepare_dataset(root/"data", downloader=downloader, expected=SMALL_SPEC)
            self.assertTrue(result["complete"])
            self.assertEqual(downloader.call_args_list, [
                mock.call("DIV2K_valid_HR/DIV2K_valid_HR/0801.png"), mock.call(),
            ])

    def test_failed_download_preserves_existing_valid_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = make_dataset(root/"data", {"0001": SMALL_SPEC["0001"]})
            before = paths["0001"].read_bytes()
            with self.assertRaises(RuntimeError):
                prepare_dataset(root/"data", downloader=mock.Mock(side_effect=RuntimeError("offline")), expected=SMALL_SPEC)
            self.assertEqual(paths["0001"].read_bytes(), before)
            self.assertFalse((root/".data.div2k.json").exists())

    def test_unrelated_images_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)/"my photos"
            destination.mkdir()
            photo = destination/"holiday.png"
            Image.new("RGB", (256, 256), "red").save(photo)
            before = photo.read_bytes()
            downloader = mock.Mock()
            with self.assertRaises(ValueError):
                prepare_dataset(destination, downloader=downloader, expected=SMALL_SPEC)
            downloader.assert_not_called()
            self.assertEqual(photo.read_bytes(), before)

    def test_duplicate_image_identifiers_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            paths = make_dataset(destination, SMALL_SPEC)
            other = destination/"DIV2K_train_HR"/"0001.png"
            other.write_bytes(paths["0001"].read_bytes())
            with self.assertRaises(ValueError):
                inventory(destination, SMALL_SPEC)


if __name__ == "__main__":
    unittest.main()
