"""Temporal export contract: global clip phase, bounded inputs, cancellation."""
import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

package = types.ModuleType('mmx_export_tests')
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package.__name__] = package
export = importlib.import_module('mmx_export_tests.director.lipsync_export')


class ExportTests(unittest.TestCase):
    def setUp(self):
        mm = types.ModuleType('comfy.model_management')
        mm.throw_exception_if_processing_interrupted = lambda: None
        self.mm = mm
        modules = patch.dict(sys.modules, {'comfy.model_management': mm})
        modules.start()
        self.addCleanup(modules.stop)

    def test_windowing_preserves_global_tokens_and_terminal_drop(self):
        # Each token depends on the complete local clip, exposing bad phase,
        # missing lookahead, misplaced terminal padding, or per-window drops.
        def encode(frames):
            blocks = []
            for start in range(0, len(frames), 17):
                clip = frames[start:start + 17, 0, 0, 0]
                clip = torch.cat((clip, clip[-1:].expand(17 - len(clip))))
                tokens = clip.mean() + torch.arange(5) + clip[0] * 10
                blocks.append(tokens)
            return torch.cat(blocks)[:-3].view(1, 1, -1, 1, 1).expand(1, 24, -1, 2, 2)

        for count in (5, 22, 56, 73, 90, 141, 294, 617, 1025):
            frames = torch.arange(count).float().view(-1, 1, 1, 1).expand(-1, 32, 32, 3)
            windows = []

            def bounded_encode(piece):
                self.assertLessEqual(len(piece), 73)
                windows.append(len(piece))
                return encode(piece)

            actual = export.encode_stitched_frames(frames, types.SimpleNamespace(encode=bounded_encode), 32, 32)
            torch.testing.assert_close(actual, encode(frames), rtol=0, atol=0)
            self.assertEqual(actual.device.type, 'cpu')
            self.assertTrue(actual.is_contiguous())
            self.assertTrue(windows)

    def test_cancellation_between_export_windows(self):
        calls = []
        def encode(frames):
            calls.append(len(frames))
            return torch.zeros(1, 24, 22, 2, 2)
        with patch.object(self.mm, 'throw_exception_if_processing_interrupted',
                          side_effect=[None, InterruptedError('cancel')]):
            with self.assertRaises(InterruptedError):
                export.encode_stitched_frames(torch.zeros(141, 32, 32, 3), types.SimpleNamespace(encode=encode), 32, 32)
        self.assertEqual(calls, [73])

    def test_invalid_grid_or_encoder_fails(self):
        vae = types.SimpleNamespace(encode=lambda frames: torch.zeros(1, 24, 1, 2, 2))
        with self.assertRaisesRegex(ValueError, 'frame grid'):
            export.encode_stitched_frames(torch.zeros(74, 32, 32, 3), vae, 32, 32)
        with self.assertRaisesRegex(RuntimeError, 'expected'):
            export.encode_stitched_frames(torch.zeros(73, 32, 32, 3), vae, 32, 32)
