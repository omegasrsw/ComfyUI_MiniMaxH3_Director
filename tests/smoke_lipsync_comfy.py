"""Integration smoke using real installed ComfyUI APIs, without diffusion weights.

Usage: python tests/smoke_lipsync_comfy.py /path/to/ComfyUI
Optional second argument: path to H3 audio VAE weights (actually encodes audio).
CLIP, video VAE and diffusion sampling use small test doubles; conditioning,
audio resampling, nested latents, context patches and PackedLayout are real.
"""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

root = Path(__file__).resolve().parents[1]
comfy_root = Path(sys.argv[1]).resolve()
audio_weights = sys.argv[2] if len(sys.argv) > 2 else None
sys.argv = sys.argv[:1]  # ComfyUI parses process arguments during imports.
sys.path.insert(0, str(comfy_root))

import torch
import comfy.ldm.minimax.model as mm
import comfy.utils
import comfy.sd

spec = importlib.util.spec_from_file_location("mmx_smoke", root / "__init__.py", submodule_search_locations=[str(root)])
plugin = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin
spec.loader.exec_module(plugin)
ls = importlib.import_module("mmx_smoke.director.lipsync")
mc = importlib.import_module("mmx_smoke.director.h3_motion_context")
node = plugin.NODE_CLASS_MAPPINGS['MiniMaxH3DirectorLongAudioLipSync']
assert node.INPUT_TYPES()['required']['audio'] == ('AUDIO',)
for path in (root / 'example_workflows').glob('*long_audio_lipsync*.json'):
    graph = json.loads(path.read_text(encoding='utf-8'))
    gen = next(n for n in graph['nodes'] if n['type'] == 'MiniMaxH3DirectorLongAudioLipSync')
    values = iter(gen['widgets_values'])
    schema = node.INPUT_TYPES()
    for name, spec in {**schema['required'], **schema['optional']}.items():
        typ = spec[0]
        if isinstance(typ, list) or typ in ('INT', 'FLOAT', 'STRING', 'BOOLEAN'):
            value = next(values)
            if isinstance(typ, list):
                assert value in typ, (name, value)
            elif typ == 'INT':
                assert type(value) is int, (name, value)
            elif typ == 'FLOAT':
                assert isinstance(value, (float, int)), (name, value)
            elif typ == 'STRING':
                assert isinstance(value, str), (name, value)
            elif typ == 'BOOLEAN':
                assert isinstance(value, bool), (name, value)
            if name == 'seed':
                assert next(values) == 'fixed'
    assert list(values) == []


class Clip:
    def tokenize(self, text, images):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 7, 8), {}]]


class VideoVAE:
    def encode(self, frames):
        return torch.zeros(1, 24, 1 if len(frames) == 1 else mc.steps_for_frames(len(frames)), 2, 2)

    def decode(self, latent):
        return torch.zeros(mc.pixel_frames_for_latent_t(latent.shape[2]), 32, 32, 3)


class AudioVAE:
    audio_sample_rate = 32000

    def encode(self, samples):
        assert samples.shape[2] == 2
        return torch.ones(1, 32, 2, (samples.shape[1] + 799) // 800)


audio_vae = AudioVAE()
if audio_weights:
    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(audio_weights, safe_load=True))

calls = []


def sample(**kwargs):
    video, audio = kwargs['latent']['samples'].unbind()
    vm, am = kwargs['latent']['noise_mask'].unbind()
    assert (vm == 1).all() and (am == 0).all()
    assert torch.isfinite(audio).all()
    positive = kwargs['positive'][0][1]
    keyframes = positive.get('minimax_keyframes', [])
    layout = mm.PackedLayout(7, video.shape[2], video.shape[3], video.shape[4], audio.shape[-1],
                             keyframes=keyframes)
    if calls:
        assert len(keyframes) == mc.steps_for_frames(22)
        positions = [float(layout.position_ids[a, 0]) for a, b, kind in layout.segments if kind == 'cond']
        assert all(a < b for a, b in zip(positions, positions[1:]))
        assert 'minimax_refs' not in positive  # no generated soundtrack substituted for source
    calls.append((video.shape, audio.shape))
    return kwargs['latent']


audio = {'waveform': torch.sin(torch.arange(44100 * 12 + 11) * 0.02).reshape(1, 1, -1), 'sample_rate': 44100}
with patch.object(ls, 'sample_single_stage', side_effect=sample):
    images, output_audio, fps, count, report = node().execute(
        model=None, clip=Clip(), video_vae=VideoVAE(), audio_vae=audio_vae, audio=audio,
        first_frame=torch.zeros(1, 32, 32, 3), prompt='A person speaking.', width=32, height=32,
        chunk_seconds=5.17, clear_vram_between_chunks=False,
    )
assert count == 289 and len(images) == 289 and fps == 24
assert output_audio is audio and len(calls) == 3
print(json.dumps({'status': 'passed', 'chunks': len(calls), 'frames': count,
                  'real_audio_vae': bool(audio_weights), 'latent_shapes': calls}))
