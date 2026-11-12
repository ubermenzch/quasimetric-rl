"""Export privacy checks that preserve scientific records and attribution."""

import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from package_nqrl_paper import anonymous_bytes, write_anonymous_archive


class PaperAnonymizationTests(unittest.TestCase):
    def test_csv_numbers_and_original_file_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'runs.csv'
            source = io.StringIO()
            writer = csv.writer(source)
            writer.writerow(['success', 'loss', 'source'])
            writer.writerow(['0.86154', '1.234567890123e-12', str(Path.home() / 'logs/test.log')])
            original = source.getvalue().encode()
            path.write_bytes(original)
            result = anonymous_bytes(path)
            row = list(csv.DictReader(io.StringIO(result.decode())))[0]
            self.assertEqual(row['success'], '0.86154')
            self.assertEqual(row['loss'], '1.234567890123e-12')
            self.assertEqual(row['source'], 'artifacts/logs/test.log')
            self.assertEqual(path.read_bytes(), original)

    def test_escaped_windows_json_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'audit.json'
            value = '\\'.join(['C:', 'Users', 'example_account', 'results', 'test.log'])
            path.write_text(json.dumps({'source': value, 'mean': 0.424}))
            result = json.loads(anonymous_bytes(path))
            self.assertEqual(result, {'source': 'artifacts\\results\\test.log', 'mean': 0.424})

    def test_private_host_and_public_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'README.md'
            host = '.'.join(['10', '82', '1', '226'])
            attribution = 'Copyright: Third-party Author <author@example.org>\nhttps://github.com/public/project'
            path.write_text(host + '\n' + attribution)
            self.assertEqual(anonymous_bytes(path).decode(), 'PRIVATE_HOST\n' + attribution)

    def test_binary_assets_remain_byte_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'figure.pdf'
            original = b'%PDF-1.4\n\x00\xff'
            path.write_bytes(original)
            self.assertEqual(anonymous_bytes(path), original)

    def test_archive_is_anonymous_with_verifiable_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'audit.json'
            source.write_text(json.dumps({'path': str(Path.home() / 'test.log'), 'success': 0.424}))
            original = source.read_bytes()
            destination = root / 'paper.zip'
            write_anonymous_archive(destination, [source], root)
            with ZipFile(destination) as archive:
                report = json.loads(archive.read('ANONYMIZATION_REPORT.json'))
                data = archive.read('audit.json')
                self.assertEqual(report['changed_files'], ['audit.json'])
                self.assertEqual(report['sha256']['audit.json'], hashlib.sha256(data).hexdigest())
                self.assertNotIn(str(Path.home()), data.decode())
                self.assertEqual(archive.getinfo('audit.json').date_time, (2026, 1, 1, 0, 0, 0))
            self.assertEqual(source.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
