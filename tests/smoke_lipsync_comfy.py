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
ref_node = plugin.NODE_CLASS_MAPPINGS['MiniMaxH3DirectorRefAudioLipSync']
assert node.INPUT_TYPES()['required']['audio'] == ('AUDIO',)
for path in (root / 'example_workflows').glob('*long_audio_lipsync*.json'):
    graph = json.loads(path.read_text(encoding='utf-8'))
    gen = next(n for n in graph['nodes'] if n['type'] in ('MiniMaxH3DirectorLongAudioLipSync', 'MiniMaxH3DirectorRefAudioLipSync'))
    values = iter(gen['widgets_values'])
    schema = plugin.NODE_CLASS_MAPPINGS[gen['type']].INPUT_TYPES()
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
    def tokenize(self, text, images=None, minimax_ref_items=None):
        if minimax_ref_items is not None:
            assert len(minimax_ref_items) == 2
            assert all(r['type'] == 'image' for r in minimax_ref_items)
        else:
            assert len(images) == 1  # original portrait remains present in every chunk
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 7, 8), {}]]


class VideoVAE:
    def encode(self, frames):
        return torch.full((1, 24, 1 if len(frames) == 1 else mc.steps_for_frames(len(frames)), 2, 2),
                          float(frames.mean()) if len(frames) == 1 else 0.0)

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
mode = 'fl2va'
original_anchor = None


def sample(**kwargs):
    global original_anchor
    video, audio = kwargs['latent']['samples'].unbind()
    vm, am = kwargs['latent']['noise_mask'].unbind()
    assert (vm == 1).all() and (am == 0).all()
    assert torch.isfinite(audio).all()
    positive = kwargs['positive'][0][1]
    keyframes = positive.get('minimax_keyframes', [])
    refs = positive.get('minimax_refs', [])
    assert len(refs) == (2 if mode == 'ref2va' else 0)
    assert all(r['kind'] == 'image' for r in refs)
    if not calls and mode == 'fl2va':
        original_anchor = keyframes[0]['latent'].clone()
    layout = mm.PackedLayout(7, video.shape[2], video.shape[3], video.shape[4], audio.shape[-1],
                             keyframes=keyframes, refs=refs)
    if calls:
        assert len(keyframes) == mc.steps_for_frames(22)
        expected_anchor = original_anchor if mode == 'fl2va' else torch.full_like(keyframes[0]['latent'], 0.3)
        torch.testing.assert_close(keyframes[0]['latent'], expected_anchor)
        assert all(k['latent'].count_nonzero() == 0 for k in keyframes[1:])
        positions = [float(layout.position_ids[a, 0]) for a, b, kind in layout.segments if kind == 'cond']
        assert all(a < b for a, b in zip(positions, positions[1:]))
        # Context time must shift with reference blocks, and both must reach
        # the real H3 payload without either list overwriting the other.
        origin = float(layout.position_ids[layout.segments[-1][0], 0])
        for position, keyframe in zip(positions, keyframes):
            assert abs(position - origin - mm.FRAME_RESCALE * keyframe['director_context_index']) < 1e-8
        import comfy.model_base
        shell = comfy.model_base.MiniMaxH3.__new__(comfy.model_base.MiniMaxH3)
        torch.nn.Module.__init__(shell)
        shell.latent_shapes = None
        with patch.object(comfy.model_base.BaseModel, 'extra_conds', return_value={}):
            payload = shell.extra_conds(**positive)['minimax_payload'].cond
        expected = [kf['latent'] for kf in keyframes] + [r['latent'] for r in refs]
        assert len(payload['cond_video_latents']) == len(expected)
        assert all(a is b for a, b in zip(payload['cond_video_latents'], expected))
        assert not payload.get('cond_audio_latents')  # source audio is in the locked target latent
    calls.append((video.shape, audio.shape))
    return kwargs['latent']


audio = {'waveform': torch.sin(torch.arange(44100 * 12 + 11) * 0.02).reshape(1, 1, -1), 'sample_rate': 44100}
with patch.object(ls, 'sample_single_stage', side_effect=sample):
    images, output_audio, fps, count, report, full_latent, positive, negative = node().execute(
        model=None, clip=Clip(), video_vae=VideoVAE(), audio_vae=audio_vae, audio=audio,
        first_frame=torch.full((1, 32, 32, 3), 0.7), prompt='A person speaking.', width=32, height=32,
        chunk_seconds=5.17, clear_vram_between_chunks=False,
        color_stabilization=0.35, detail_stabilization=0.35,
        first_frame_anchor_strength=1.0,
    )
assert count == 289 and len(images) == 289 and fps == 24
assert output_audio is audio and len(calls) == 3
assert full_latent['samples'].shape[2] == mc.steps_for_frames(ls.minimax_align_frame_count(count))
assert not positive[0][1].get('minimax_keyframes')
assert negative == []
print(json.dumps({'status': 'passed', 'chunks': len(calls), 'frames': count,
                  'mode': mode, 'real_audio_vae': bool(audio_weights), 'latent_shapes': calls}))
calls.clear()
mode = 'ref2va'
before = audio['waveform'].clone()
with patch.object(ls, 'sample_single_stage', side_effect=sample):
    images, output_audio, fps, count, report, full_latent, positive, negative = ref_node().execute(
        model=None, clip=Clip(), video_vae=VideoVAE(), audio_vae=audio_vae, audio=audio,
        ref_image_1=torch.zeros(1, 32, 32, 3), ref_image_2=torch.ones(1, 32, 32, 3),
        prompt='The character in <Picture 1> in the background from <Picture 2>.',
        width=32, height=32, chunk_seconds=5.17, clear_vram_between_chunks=False,
        appearance_reference=torch.full((1, 32, 32, 3), 0.3),
        color_stabilization=0.35, detail_stabilization=0.35,
        first_frame_anchor_strength=1.0,
    )
assert count == len(images) == 289 and fps == 24 and len(calls) == 3
assert output_audio is audio and torch.equal(output_audio['waveform'], before)
assert json.loads(report)['source_audio_locked']
assert json.loads(report)['context_source'] == 'corrected decoded tail'
assert not full_latent['samples'].is_nested
assert full_latent['samples'].shape[2] == mc.steps_for_frames(ls.minimax_align_frame_count(count))
assert len(positive[0][1]['minimax_refs']) == 2 and not positive[0][1].get('minimax_keyframes')
assert positive[0][1]['minimax_frame_count'] == ls.minimax_align_frame_count(count)
assert negative == []
print(json.dumps({'status': 'passed', 'mode': mode, 'chunks': len(calls), 'frames': count,
                  'real_audio_vae': bool(audio_weights), 'audio_unchanged': True,
                  'context_source': json.loads(report)['context_source'],
                  'whole_video_latent_shape': list(full_latent['samples'].shape)}))
