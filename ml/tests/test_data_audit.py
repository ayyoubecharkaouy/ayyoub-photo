"""Contrôles d'audit sur de petits fichiers artificiels, jamais le dataset utilisateur."""

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('ayyphoto_data_audit', ROOT / 'ml/data_audit.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class DataAuditTests(unittest.TestCase):
    def test_current_split_is_disjoint_and_matches_notebook_counts(self):
        splits = {str(i): audit.identify_split(f'DIV2K_train_HR/{i:04d}.png') for i in range(1, 801)}
        self.assertEqual(list(splits.values()).count('train'), 720)
        self.assertEqual(list(splits.values()).count('val'), 80)
        self.assertTrue(all(audit.identify_split(f'DIV2K_valid_HR/{i:04d}.png') == 'test' for i in range(801, 901)))
        self.assertEqual(audit.identify_split('other/not-an-id.png'), 'unknown')

    def test_decode_rejects_corruption_and_transparency(self):
        with self.assertRaises(Exception):
            audit.decode_image(b'not an image')
        import io
        stream = io.BytesIO()
        Image.new('RGBA', (10, 10)).save(stream, format='PNG')
        with self.assertRaises(ValueError):
            audit.decode_image(stream.getvalue())

    def test_phash_candidates_are_not_exact_duplicates(self):
        records = [
            {'path': 'a', 'split': 'train', 'pixel_sha256': 'x', 'phash63': '0000000000000001'},
            {'path': 'b', 'split': 'test', 'pixel_sha256': 'y', 'phash63': '0000000000000003'},
        ]
        thumbnails = {'a': np.full((48, 48, 3), 100, dtype=np.uint8),
                      'b': np.full((48, 48, 3), 102, dtype=np.uint8)}
        candidates, sampled = audit.near_duplicate_candidates(records, thumbnails)
        self.assertFalse(sampled)
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]['cross_split'])
        self.assertEqual(candidates[0]['hash_distance'], 1)

    def test_full_audit_reports_errors_duplicates_and_preserves_sources(self):
        with tempfile.TemporaryDirectory(prefix='ayyphoto-data-audit-') as temporary:
            root = Path(temporary)
            data, output = root / 'data', root / 'reports'
            train, test = data / 'DIV2K_train_HR', data / 'DIV2K_valid_HR'
            train.mkdir(parents=True)
            test.mkdir()
            pixels = np.random.default_rng(42).integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(train / '0001.png')
            (train / '0002.png').write_bytes((train / '0001.png').read_bytes())
            (test / '0801.png').write_bytes((train / '0001.png').read_bytes())
            Image.new('RGB', (20, 20), 'white').save(train / '0003.png')
            (train / 'broken.png').write_bytes(b'broken PNG')
            before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in data.rglob('*') if p.is_file()}
            report = audit.audit_dataset(data, output)
            self.assertEqual(report['files_found'], 5)
            self.assertEqual(report['decoded_images'], 4)
            self.assertEqual(report['errors_count'], 1)
            self.assertEqual(report['exact_pixel_duplicate_groups'], 1)
            self.assertEqual(report['cross_split_exact_groups'], 1)
            self.assertEqual(report['too_small_count'], 1)
            self.assertFalse(report['compatible_with_current_900_image_loader'])
            self.assertEqual(report['content_quality_status'], 'manual_review_required')
            for name in ('summary.json', 'images.csv', 'errors.csv', 'near_duplicates.csv',
                         'exact_duplicates.json', 'sample.jpg', 'review_candidates.jpg'):
                self.assertTrue((output / name).is_file(), name)
            self.assertEqual(before, {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in data.rglob('*') if p.is_file()})

    def test_audit_does_not_write_into_source_or_overwrite_report(self):
        with tempfile.TemporaryDirectory(prefix='ayyphoto-audit-location-') as temporary:
            data = Path(temporary) / 'data'
            data.mkdir()
            Image.new('RGB', (256, 256)).save(data / '0001.png')
            with self.assertRaises(ValueError):
                audit.audit_dataset(data, data / 'reports')
            output = Path(temporary) / 'existing-report'
            output.mkdir()
            (output / 'keep.txt').write_text('keep')
            with self.assertRaises(FileExistsError):
                audit.audit_dataset(data, output)
            self.assertEqual((output / 'keep.txt').read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
