"""CPU regression tests; run python -m unittest discover -s tests -v."""

import importlib
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

# Import helpers without executing the ComfyUI plugin registration entry point.
ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("mmx_test_package")
package.__path__ = [str(ROOT)]
sys.modules.setdefault(package.__name__, package)
ls = importlib.import_module("mmx_test_package.director.lipsync")
mc = importlib.import_module("mmx_test_package.director.h3_motion_context")


class TimelineTests(unittest.TestCase):
    def test_coverage_and_phase_alignment(self):
        for rate in (16000, 32000, 44100, 48000):
            for samples in (1, rate * 5, rate * 15 + 1, rate * 61 + 137, rate * 3600 + 1):
                for context in (5, 22, 39, 56):
                    for seconds in (5.17, 10, 15):
                        chunks = ls.plan_audio_chunks(samples, rate, seconds, context)
                        expected = (samples * 24 + rate - 1) // rate
                        self.assertEqual(chunks[-1].end_frame, expected)
                        self.assertEqual(sum(c.visible_frames for c in chunks), expected)
                        for c in chunks:
                            self.assertGreater(c.visible_frames, 0)
                            self.assertEqual(c.sample_frames % 17, 5)
                            self.assertLessEqual(c.sample_frames, int(seconds * 24))
                            self.assertGreaterEqual(c.sample_frames, 124)
                            self.assertGreaterEqual(c.audio_start_frame, 0)
                        for prev, current in zip(chunks, chunks[1:]):
                            self.assertEqual(prev.end_frame, current.start_frame)
                            self.assertEqual(prev.sample_frames, prev.context_frames + prev.visible_frames)
                            steps = mc.steps_for_frames(prev.sample_frames)
                            start, end, gap = mc._phase_aligned_tail_start(steps, mc.steps_for_frames(context), None)
                            self.assertEqual(gap, 0)
                            self.assertEqual(start % 5, 0)
                            self.assertEqual(end, prev.sample_frames)

    def test_audio_slices_use_absolute_clock_and_eof_padding(self):
        for rate in (16000, 32000, 44100, 48000):
            samples = rate * 27 + 7
            audio = {"waveform": torch.arange(samples).reshape(1, 1, -1).float(), "sample_rate": rate}
            for c in ls.plan_audio_chunks(samples, rate):
                sliced = ls.slice_chunk_audio(audio, c)["waveform"]
                start = ls.sample_offset(c.audio_start_frame, rate)
                end = ls.sample_offset(c.audio_start_frame + c.sample_frames, rate)
                n = min(end, samples) - start
                self.assertEqual(sliced.shape, (1, 2, end - start))
                torch.testing.assert_close(sliced[:, :1, :n], audio["waveform"][..., start:start + n])
                torch.testing.assert_close(sliced[:, 0], sliced[:, 1])
                self.assertEqual(sliced[..., n:].count_nonzero(), 0)
                boundary = ls.sample_offset(c.start_frame, rate) - start
                self.assertEqual(sliced[0, 0, boundary].item(), ls.sample_offset(c.start_frame, rate))

    def test_bad_audio_and_settings(self):
        for samples, rate, seconds, context in ((0, 32000, 10, 22), (1, 0, 10, 22),
                (1, 32000, 16, 22), (1, 32000, 5, 22), (1, 32000, 10, 7), (1, 32000, float('nan'), 22)):
            with self.assertRaises(ValueError):
                ls.plan_audio_chunks(samples, rate, seconds, context)
        for waveform in (torch.zeros(1, 0), torch.zeros(2, 2, 1), torch.zeros(1, 3, 1),
                         torch.zeros(1, 2, 0), torch.full((1, 2, 1), float('nan'))):
            with self.assertRaises(ValueError):
                ls.validate_audio({"waveform": waveform, "sample_rate": 32000})

    def test_prompts_fail_before_sampling(self):
        self.assertEqual(ls.parse_chunk_prompts('', 2, 'base'), ['base', 'base'])
        self.assertEqual(ls.parse_chunk_prompts('["a", "b"]', 2, 'base'), ['base\na', 'base\nb'])
        for raw in ('{}', '["one"]', '[1, 2]', 'oops'):
            with self.assertRaises(ValueError):
                ls.parse_chunk_prompts(raw, 2, 'base')

    def test_audio_mask_time_axis(self):
        chunks = ls.plan_audio_chunks(32000 * 27, 32000)
        mask = torch.arange(648).float().reshape(-1, 1, 1) / 648
        c = chunks[1]
        target = torch.zeros(1, 32, 2, 377)
        out = ls.audio_mask_for_chunk(mask, c, target, 648, 0)
        self.assertEqual(out.shape, target.shape)
        self.assertEqual(out[0, 0, 0, 0], mask[c.audio_start_frame, 0, 0])
        torch.testing.assert_close(out[:, :, 0], out[:, :, 1])
        self.assertEqual(ls.audio_mask_for_chunk(None, c, target, 648, 0).count_nonzero(), 0)
        with self.assertRaises(ValueError):
            ls.audio_mask_for_chunk(torch.zeros(24, 8, 8), c, target, 648, 0)


class Nested:
    def __init__(self, items):
        self.items = items

    def unbind(self):
        return self.items


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.samples = []
        self.encoded_audio = []
        self.contexts = []
        self.cleanups = []

        def conditioning(**kwargs):
            n = kwargs['length']
            self.assertEqual(kwargs['first_frame'] is None, len(self.samples) > 0)
            return ['positive'], {'samples': Nested((torch.zeros(1, 24, mc.steps_for_frames(n), 2, 2),
                                                   torch.zeros(1, 32, 2, round(n / 24 * 40))))}

        def encode(vae, audio):
            self.encoded_audio.append(audio)
            return ({'samples': torch.ones(1, 32, 2, round(audio['waveform'].shape[-1] / audio['sample_rate'] * 40))},)

        def sample(**kwargs):
            video, audio = kwargs['latent']['samples'].unbind()
            vm, am = kwargs['latent']['noise_mask'].unbind()
            self.assertEqual(am.count_nonzero(), 0)
            self.assertTrue((vm == 1).all())
            self.assertTrue((audio == 1).all())
            self.samples.append(kwargs)
            # Mark time within the generated chunk to expose accidental prefix retention.
            return {'samples': Nested((video, audio)), 'index': len(self.samples) - 1}

        def decode(vae, latent):
            n = mc.pixel_frames_for_latent_t(latent['samples'].unbind()[0].shape[2])
            return (torch.arange(n).reshape(-1, 1, 1, 1).expand(n, 2, 2, 3).float(),)

        def context(positive, latent, **kwargs):
            self.contexts.append(kwargs)
            # Exercise actual phase-aligned extraction on the compact retained tail.
            _, _, covered, _, gap = mc._video_tail_blocks(kwargs['context_latent'], kwargs['context_length'])
            self.assertFalse(kwargs['continue_audio'])
            return positive, covered, gap

        def module(name, **attrs):
            mod = types.ModuleType(name)
            mod.__dict__.update(attrs)
            return mod

        mm = module('comfy.model_management', throw_exception_if_processing_interrupted=lambda: None)
        mods = {
            'comfy': module('comfy', model_management=mm),
            'comfy.model_management': mm,
            'comfy.nested_tensor': module('comfy.nested_tensor', NestedTensor=Nested),
            'comfy.utils': module('comfy.utils', ProgressBar=lambda n: types.SimpleNamespace(update_absolute=lambda i: None)),
            'comfy_extras': module('comfy_extras'),
            'comfy_extras.nodes_minimax_h3': module('comfy_extras.nodes_minimax_h3', MiniMaxH3ImageToVideo=types.SimpleNamespace(execute=conditioning)),
            'comfy_extras.nodes_audio': module('comfy_extras.nodes_audio', VAEEncodeAudio=types.SimpleNamespace(execute=encode)),
            'nodes': module('nodes', VAEDecode=lambda: types.SimpleNamespace(decode=decode)),
        }
        self.patches = [patch.dict(sys.modules, mods), patch.object(ls, 'sample_single_stage', side_effect=sample),
                        patch.object(ls, 'apply_motion_context', side_effect=context),
                        patch.object(ls, 'cleanup_segment_vram', side_effect=lambda **k: self.cleanups.append(k))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def run_generation(self, samples=32000 * 27 + 7, **kwargs):
        self.audio = {'waveform': torch.ones(1, 1, samples), 'sample_rate': 32000}
        return ls.generate_lipsync(model=None, clip=None, video_vae=None, audio_vae=None,
            audio=self.audio, first_frame=torch.ones(1, 32, 32, 3), prompt='speech', width=32, height=32,
            seed=2**64-1, **kwargs)

    def test_multichunk_exact_output_and_overlap_removal(self):
        images, audio, fps, count, report = self.run_generation()
        self.assertIs(audio, self.audio)
        self.assertEqual(count, 649)
        self.assertEqual(images.shape[0], count)
        self.assertEqual(fps, 24)
        chunks = json.loads(report)['chunks']
        self.assertEqual(len(self.contexts), len(chunks) - 1)
        for c in chunks:
            self.assertEqual(images[c['start_frame'], 0, 0, 0], c['context_frames'])
            self.assertEqual(images[c['end_frame'] - 1, 0, 0, 0], c['context_frames'] + c['end_frame'] - c['start_frame'] - 1)
        self.assertEqual([s['seed'] for s in self.samples[:2]], [2**64-1, 0])
        self.assertEqual(len(self.cleanups), len(chunks))

    def test_one_sample_audio(self):
        images, _, _, count, report = self.run_generation(samples=1)
        self.assertEqual(count, 1)
        self.assertEqual(images.shape[0], 1)
        self.assertEqual(len(self.contexts), 0)
        self.assertEqual(json.loads(report)['chunks'][0]['sample_frames'], 124)

    def test_audio_vae_rounding_and_wrong_grid(self):
        encoder = sys.modules['comfy_extras.nodes_audio'].VAEEncodeAudio
        for tokens in (206, 208):  # 124 video frames expect 207 audio tokens
            self.samples.clear()
            with patch.object(encoder, 'execute', return_value=({'samples': torch.ones(1, 32, 2, tokens)},)):
                self.run_generation(samples=1)
                self.assertEqual(self.samples[-1]['latent']['samples'].unbind()[1].shape[-1], 207)
        self.samples.clear()
        with patch.object(encoder, 'execute', return_value=({'samples': torch.ones(1, 32, 2, 100)},)):
            with self.assertRaisesRegex(ValueError, 'grid mismatch'):
                self.run_generation(samples=1)

    def test_short_decode_is_error(self):
        with patch.object(sys.modules['nodes'], 'VAEDecode', return_value=types.SimpleNamespace(decode=lambda *a: (torch.zeros(1, 2, 2, 3),))):
            with self.assertRaisesRegex(RuntimeError, 'decoded'):
                self.run_generation()

    def test_cancellation_stops_before_next_sample(self):
        mm = sys.modules['comfy.model_management']
        with patch.object(mm, 'throw_exception_if_processing_interrupted', side_effect=[None, InterruptedError('cancelled')]):
            with self.assertRaises(InterruptedError):
                self.run_generation()
        self.assertEqual(len(self.samples), 1)


class WorkflowTests(unittest.TestCase):
    def test_examples_have_reciprocal_links_and_original_audio_output(self):
        for path in (ROOT / 'example_workflows').glob('*long_audio_lipsync*.json'):
            graph = json.loads(path.read_text(encoding='utf-8'))
            nodes = {n['id']: n for n in graph['nodes']}
            links = {l[0]: l for l in graph['links']}
            self.assertEqual(len(nodes), len(graph['nodes']))
            self.assertEqual(len(links), len(graph['links']))
            for lid, src, out, dst, inp, typ in links.values():
                self.assertIn(lid, nodes[src]['outputs'][out]['links'])
                self.assertEqual(nodes[dst]['inputs'][inp]['link'], lid)
                self.assertEqual(nodes[src]['outputs'][out]['type'], typ)
                self.assertIn(nodes[dst]['inputs'][inp]['type'], (typ, '*'))
            for n in nodes.values():
                for i in n['inputs']:
                    if i['link'] is not None:
                        self.assertIn(i['link'], links)
                for o in n['outputs']:
                    for lid in o.get('links') or []:
                        self.assertIn(lid, links)
            gen = next(n for n in nodes.values() if n['type'] == 'MiniMaxH3DirectorLongAudioLipSync')
            create = next(n for n in nodes.values() if n['type'] == 'CreateVideo')
            audio_link = next(i['link'] for i in create['inputs'] if i['name'] == 'audio')
            self.assertEqual(links[audio_link][1:3], [gen['id'], 1])
            values = gen['widgets_values']
            self.assertEqual(values[3:5], [10.0, 22])
            self.assertEqual(values[6], 'fixed')
            self.assertEqual(values[12], 0.0)  # lock source audio
            self.assertEqual(values[-1], '')  # optional chunk prompts


if __name__ == '__main__':
    unittest.main()
