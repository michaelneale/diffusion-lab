"""Offline checks only: no model loads, native workers, SSH, or generated outputs."""
import ast
import importlib.util
import json
import os
import runpy
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LLADA = ROOT / 'llada-split-20261117'
sys.path.insert(0, str(LLADA))
import canvas_experiment as canvas
import probe32_experiment as probe
import wan_sweep as sweep
import wan_fresh as fresh

spec = importlib.util.spec_from_file_location('wire', ROOT / 'lossy-diffusion-20261118/wire.py')
wire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wire)


class SamplingTests(unittest.TestCase):
    def test_existing_selfchecks(self):
        canvas.selftest()
        sweep.checks()

    def test_ties_and_block_boundary(self):
        tokens = [7, canvas.MASK, canvas.MASK, canvas.MASK]
        self.assertEqual(canvas.choose_updates(tokens, 1, [8, 9, 10], [.8, .8, 1], 1, 1, 3), [1])
        self.assertEqual(tokens, [7, 8, canvas.MASK, canvas.MASK])

    def test_reject_mask_as_progress(self):
        with self.assertRaises(RuntimeError):
            canvas.choose_updates([7, canvas.MASK], 1, [canvas.MASK], [.9], 1, 1, 2)

    def test_nonfinite_confidence(self):
        with self.assertRaises(AssertionError):
            canvas.choose_updates([7, canvas.MASK], 1, [8], [float('nan')], 1, 1, 2)

    def test_immutable_prefix(self):
        eos = min(canvas.SPECIAL_END)
        self.assertIsNone(sweep.complete_prefix([canvas.MASK, eos]))
        self.assertEqual(sweep.complete_prefix([4, eos, canvas.MASK]), 1)

    def test_prefix_replay(self):
        record = {'visible_tokens_before_end': 1, 'canvas_tokens': 4, 'prompt_tokens': 2,
                  'passes': [{'selected_positions': [3], 'wall_s': 1},
                             {'selected_positions': [2], 'wall_s': 2}]}
        out = probe.prefix_completion(record)
        self.assertEqual(out['prefix_complete_pass'], 2)
        self.assertEqual(out['prefix_eval_wall_s'], 3)

    def test_local_command_needs_no_ssh_configuration(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(canvas.worker_command('full'), [str(LLADA / 'canvas_worker'), canvas.MODEL, 'full'])

    def test_remote_configuration_required(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'DIFFUSION_SSH_HOST'):
                canvas.worker_command('tail', True)

    def test_remote_path_quoted(self):
        with patch.dict(os.environ, {'DIFFUSION_SSH_HOST': 'lab@example.invalid',
                                     'DIFFUSION_REMOTE_DIR': 'lab files/experiment'}):
            cmd = canvas.worker_command('tail', True)
        self.assertEqual(cmd[-3:-1], ['--', 'lab@example.invalid'])
        self.assertTrue(cmd[-1].startswith("cd 'lab files/experiment' && mkdir -p home && "))
        self.assertIn('HOME="$PWD/home"', cmd[-1])

    def test_adaptive_local_does_not_start_split_workers(self):
        with patch.object(probe, 'LocalExperiment') as local, \
             patch.object(canvas, 'Experiment', side_effect=AssertionError('unexpected split')), \
             patch.object(sys, 'argv', ['adaptive_experiment.py', '--thresholds', '.8']), \
             patch('signal.signal'), patch('signal.alarm'):
            local.return_value.results = {}
            runpy.run_path(str(LLADA / 'adaptive_experiment.py'), run_name='__main__')
            local.assert_called_once()
            local.return_value.generate.assert_called_once()
            self.assertFalse(local.return_value.generate.call_args.args[4])
            local.return_value.close.assert_called_once()

    def test_gguf_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'header.gguf'
            path.write_bytes(b'GGUF' + struct.pack('<IQQ', 3, 12, 0))
            self.assertEqual(fresh.metadata(path), {'_tensor_count': 12})
            path.write_bytes(b'GGUF')
            with self.assertRaises(ValueError):
                fresh.metadata(path)


class CodecTests(unittest.TestCase):
    def test_existing_selfchecks(self):
        wire.self_test()

    def test_bf16_exact(self):
        x = np.arange(128, dtype=np.float32).reshape(1, 2, 64)
        np.testing.assert_array_equal(wire.decode(wire.encode(x, 17)), x)

    def test_bf16_rejects_non_bf16_input(self):
        with self.assertRaises(AssertionError):
            wire.encode(np.full((1, 1, 64), .1234567, np.float32), 17)

    def test_header_and_length_rejections(self):
        packet = wire.encode(np.zeros((1, 2, 64), np.float32), 8)
        for corrupt in [b'NOPE' + packet[4:], packet[:-1], packet + b'x']:
            with self.subTest(length=len(corrupt)):
                with self.assertRaises(AssertionError):
                    wire.decode(corrupt)

    def test_finite_input_required(self):
        for bits in [4, 8, 16, 17]:
            with self.subTest(bits=bits):
                with self.assertRaises(AssertionError):
                    wire.encode(np.full((1, 1, 64), np.nan, np.float32), bits)

    def test_serialized_sizes(self):
        x = np.ones((1, 3, 2816), np.float32)
        for bits in [4, 8, 16, 17]:
            expected = 16 + x.size * (16 if bits == 17 else bits) // 8
            if bits in [4, 8]:
                expected += x.size // 64 * 2
            self.assertEqual(len(wire.encode(x, bits)), expected)


class BundleTests(unittest.TestCase):
    def test_all_python_parses(self):
        paths = [*ROOT.glob('*-2026*/*.py'), *ROOT.glob('tests/*.py')]
        self.assertEqual(len(paths), 20)
        for path in paths:
            with self.subTest(path=path.name):
                ast.parse(path.read_text(), filename=str(path))

    def test_prompt_and_hash_fixtures(self):
        incident = json.loads((LLADA / 'incident_prompt.json').read_text())
        self.assertEqual(set(incident), {'prompt_token_ids'})
        self.assertEqual(len(incident['prompt_token_ids']), 89)
        self.assertTrue(all(type(x) is int for x in incident['prompt_token_ids']))
        for path in ROOT.glob('*/tasks.json'):
            tasks = json.loads(path.read_text())
            self.assertTrue(tasks)
            self.assertTrue(all(isinstance(v, str) and v for v in tasks.values()))
        hashes = json.loads((ROOT / 'lossy-diffusion-20261118/runtime_hashes.json').read_text())
        self.assertEqual(len(hashes), 3)
        for digest in hashes.values():
            self.assertEqual(len(bytes.fromhex(digest)), 32)

    def test_empty_evidence_fails_instead_of_claiming_success(self):
        suites = [('llada-split-20261117', 'wan_validate.py'),
                  ('llada-split-20261117', 'probe32_validate.py'),
                  ('lowpass-search-20261117', 'validate.py'),
                  ('nemotron-minpasses-20261117', 'validate.py')]
        for directory, script in suites:
            with self.subTest(directory=directory, script=script), tempfile.TemporaryDirectory() as temporary:
                for source in (ROOT / directory).glob('*.py'):
                    shutil.copyfile(source, Path(temporary) / source.name)
                result = subprocess.run([sys.executable, script], cwd=temporary,
                                        capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('No trial records found', result.stderr)


if __name__ == '__main__':
    unittest.main()
