import importlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import torch

from test_lipsync import ROOT  # installs the lightweight plugin package alias

export = importlib.import_module('mmx_test_package.lib.exact_audio_video')


class ExactAudioTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = ROOT / '_seam_probe'
        self.temp_root.mkdir(exist_ok=True)

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg/ffprobe required')
    def test_mkv_audio_decodes_to_identical_samples(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            for dtype, np_dtype, sample_rate, channels in ((torch.float32, '<f4', 44100, 2), (torch.float64, '<f8', 32000, 1)):
                waveform = torch.linspace(-0.37, 0.43, sample_rate + 7, dtype=dtype).reshape(1, 1, -1).repeat(1, channels, 1)
                audio = {'waveform': waveform, 'sample_rate': sample_rate}
                path = Path(folder) / f'{channels}.mkv'
                frames = torch.zeros(25, 16, 16, 3)
                export.write_exact_audio_video(path, frames, audio)
                fmt = 'f32le' if dtype == torch.float32 else 'f64le'
                raw = subprocess.check_output([shutil.which('ffmpeg'), '-v', 'error', '-i', str(path), '-map', '0:a:0', '-f', fmt, '-'])
                expected = waveform[0].transpose(0, 1).contiguous().numpy().astype(np_dtype, copy=False).tobytes()
                self.assertEqual(raw, expected)
                info = json.loads(subprocess.check_output([shutil.which('ffprobe'), '-v', 'error', '-show_streams', '-of', 'json', str(path)]))
                stream = next(s for s in info['streams'] if s['codec_type'] == 'audio')
                self.assertEqual(int(stream['sample_rate']), sample_rate)
                self.assertEqual(stream['channels'], channels)
                self.assertEqual(stream['codec_name'], 'pcm_' + fmt)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'FFmpeg required')
    def test_cancel_does_not_publish_partial_video(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as folder:
            target = Path(folder) / 'cancel.mkv'
            def cancel():
                raise InterruptedError('cancelled')
            with self.assertRaises(InterruptedError):
                export.write_exact_audio_video(target, torch.zeros(25, 16, 16, 3),
                    {'waveform': torch.zeros(1, 1, 32000), 'sample_rate': 32000}, check_interrupt=cancel)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])
