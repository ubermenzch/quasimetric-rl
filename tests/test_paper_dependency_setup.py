import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PaperDependencySetupTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        scripts = self.root / 'tools'
        scripts.mkdir()
        for name in ('setup_baseline_paper_dependencies.sh', 'qrl_env.sh'):
            shutil.copy2(ROOT / 'tools' / name, scripts / name)
        self.script = scripts / 'setup_baseline_paper_dependencies.sh'
        binaries = self.root / '.venv/bin'
        binaries.mkdir(parents=True)
        # Execute the real smoke-check driver, mocking only simulator processes.
        python = binaries / 'python'
        python.write_text(f'#!{sys.executable}\n' + '''
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

assert sys.argv[1:] == ['-'], sys.argv
program = sys.stdin.read()
calls = []
def simulate(command, **kwargs):
    assert command[:2] == [sys.executable, '-c'], command
    calls.append(command[2])
    Path(os.environ['QRL_TEST_CHECK_CALLS']).write_text(json.dumps(calls))
    code = int(os.environ.get('QRL_TEST_FAIL_ROBEL') == '1' and 'robel' in command[2])
    return subprocess.CompletedProcess(command, code)
with patch('subprocess.run', side_effect=simulate):
    exec(compile(program, '<setup-smoke-checks>', 'exec'), {'__name__': '__main__'})
''')
        python.chmod(0o755)
        curl = binaries / 'curl'
        curl.write_text('#!/bin/sh\ntouch "$QRL_TEST_NETWORK_CALLS"\nexit 99\n')
        curl.chmod(0o755)
        self.calls = self.root / 'checks.json'
        self.network_calls = self.root / 'network-called'
        self.datasets = self.root / 'datasets'
        self.env = dict(os.environ,
                        QRL_ASSET_ROOT=str(self.root / 'assets'),
                        QRL_PAPER_REFERENCE_ROOT=str(self.root / 'references'),
                        D4RL_DATASET_DIR=str(self.datasets),
                        QRL_TEST_CHECK_CALLS=str(self.calls),
                        QRL_TEST_NETWORK_CALLS=str(self.network_calls),
                        QRL_TEST_FAIL_ROBEL='0',
                        QRL_PEN_DATASET_URL='https://example.invalid/pen.hdf5')

    def run_setup(self, *args):
        result = subprocess.run(['sh', str(self.script), *args], env=self.env,
                                text=True, capture_output=True, timeout=30)
        self.assertFalse(self.network_calls.exists())
        self.assertFalse(self.datasets.exists())
        self.assertIn('[4/5] SKIPPED: C-Learning Pen', result.stdout)
        return result

    def test_resume_checks_and_legacy_alias_skip_pen_without_downloading(self):
        for option in ('--checks-only', '--from-data'):
            with self.subTest(option=option):
                result = self.run_setup(option)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertNotIn('[1/5]', result.stdout)
                self.assertNotIn('[3/5]', result.stdout)
                checks = json.loads(self.calls.read_text())
                self.assertEqual(len(checks), 4)
                self.assertFalse(any('pen-human-v0' in check for check in checks))
                for name in ('Box2D', 'ROBEL DClaw', 'D4RL Door base', 'DMC imports'):
                    self.assertIn(name + ': PASS', result.stdout)

    def test_other_simulator_failure_still_fails_setup(self):
        self.env['QRL_TEST_FAIL_ROBEL'] = '1'
        result = self.run_setup('--checks-only')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('ROBEL DClaw: FAIL', result.stdout)
        self.assertIn('DMC imports: PASS', result.stdout)
        self.assertNotIn('Dependency preparation finished.', result.stdout)

    def test_full_dry_run_omits_pen_download(self):
        result = self.run_setup('--dry-run')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('git clone', result.stdout)
        self.assertNotIn('pen-v0_demos_clipped.hdf5', result.stdout)
        self.assertNotIn('curl', result.stdout)
        self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
