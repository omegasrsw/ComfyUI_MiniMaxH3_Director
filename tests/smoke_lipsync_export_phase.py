"""Check windowed export against installed H3 encode/encode_temporal on CPU.

Usage: python tests/smoke_lipsync_export_phase.py COMFY_ROOT
Uses the real normalization/clip/padding/drop methods, with a lightweight
stand-in for the neural encoder. No model weights or GPU render required.
"""
import importlib
from pathlib import Path
import sys
import types

root = Path(__file__).resolve().parents[1]
comfy_root = sys.argv[1]
sys.argv = [sys.argv[0], '--cpu']
sys.path.insert(0, comfy_root)
import torch
from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE, LATENTS_MEAN, LATENTS_STD, IMAGENET_MEAN, IMAGENET_STD

package = types.ModuleType('mmx_phase_smoke')
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
export = importlib.import_module('mmx_phase_smoke.director.lipsync_export')


class TemporalProbe(MiniMaxH3VideoVAE):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.clip_length = 17
        self.token_drop = 3
        self.register_buffer('latents_mean', torch.tensor(LATENTS_MEAN))
        self.register_buffer('latents_std', torch.tensor(LATENTS_STD))
        self.register_buffer('pixel_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1))
        self.register_buffer('pixel_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1))

    def _adaptive_encode(self, clip):
        assert clip.shape[2] == 17
        # Every token depends on its entire clip, including its repeated tail.
        mean = clip.mean() + torch.arange(5).view(1, 1, 5, 1, 1)
        return mean.expand(1, 48, 5, 2, 2).clone()


probe = TemporalProbe()


class VAE:
    def encode(self, frames):
        x = frames.movedim(-1, 1).movedim(1, 0).unsqueeze(0)
        return probe.encode(x * 2 - 1, device=torch.device('cpu'))


with torch.inference_mode():
    for count in (5, 22, 73, 90, 141, 294, 617, 1025):
        frames = torch.rand(count, 32, 32, 3)
        actual = export.encode_stitched_frames(frames, VAE(), 32, 32)
        expected = VAE().encode(frames)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
print('Passed: windowed export equals installed H3 temporal encoding for eight timeline lengths (CPU probe).')
