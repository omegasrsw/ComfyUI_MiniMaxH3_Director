"""CPU regression tests; run python -m unittest discover -s tests -v."""

import importlib
import json
from pathlib import Path
import sys
import types
import unittest
import weakref
from unittest.mock import patch

import torch

# Import helpers without executing the ComfyUI plugin registration entry point.
ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("mmx_test_package")
package.__path__ = [str(ROOT)]
sys.modules.setdefault(package.__name__, package)
ls = importlib.import_module("mmx_test_package.director.lipsync")
mc = importlib.import_module("mmx_test_package.director.h3_motion_context")
le = importlib.import_module("mmx_test_package.director.lipsync_export")


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
        self.reference_each_chunk = True
        self.reference_calls = []
        self.conditioning_calls = []
        self.real_export = False
        actual_export = le.export_stitched_video

        def export(**kwargs):
            if self.real_export:
                return actual_export(**kwargs)
            return {'samples': torch.zeros(1, 24, mc.steps_for_frames(len(kwargs['frames'])), 2, 2)}, [], []

        def conditioning(**kwargs):
            self.conditioning_calls.append(kwargs)
            n = kwargs['length']
            self.assertEqual(kwargs['first_frame'] is None, len(self.samples) > 0 and not self.reference_each_chunk)
            metadata = {'minimax_keyframes': [{'resolved_frame_index': 0, 'latent': torch.zeros(1, 24, 1, 2, 2)}]}
            return [[torch.zeros(1, 7, 8), metadata]], {'samples': Nested((torch.zeros(1, 24, mc.steps_for_frames(n), 2, 2),
                                                   torch.zeros(1, 32, 2, round(n / 24 * 40))))}

        def encode(vae, audio):
            self.encoded_audio.append(audio)
            return ({'samples': torch.ones(1, 32, 2, round(audio['waveform'].shape[-1] / audio['sample_rate'] * 40))},)

        def reference_conditioning(**kwargs):
            self.reference_calls.append(kwargs)
            self.assertNotIn('first_frame', kwargs)
            self.assertNotIn('ref_audios', kwargs)
            n = kwargs['length']
            refs = [{'kind': 'image', 'slot': k} for k in kwargs['ref_images']]
            return [[torch.zeros(1, 7, 8), {'minimax_refs': refs}]], {'samples': Nested((
                torch.zeros(1, 24, mc.steps_for_frames(n), 2, 2), torch.zeros(1, 32, 2, round(n / 24 * 40))))}

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
            self.assertEqual(set(latent), {'samples'})
            self.assertIsInstance(latent['samples'], torch.Tensor)
            self.assertEqual(latent['samples'].device.type, 'cpu')
            n = mc.pixel_frames_for_latent_t(latent['samples'].shape[2])
            return (torch.arange(n).reshape(-1, 1, 1, 1).expand(n, 2, 2, 3).float(),)

        def context(positive, latent, **kwargs):
            self.contexts.append(kwargs)
            # Exercise actual phase-aligned extraction on the compact retained tail.
            blocks, offsets, covered, _, gap = mc._video_tail_blocks(kwargs['context_latent'], kwargs['context_length'])
            self.assertFalse(kwargs['continue_audio'])
            positive = [[embedding, {**metadata, 'minimax_keyframes': [
                {'resolved_frame_index': 0, 'director_context_index': offset, 'latent': block}
                for offset, block in zip(offsets, blocks)]}] for embedding, metadata in positive]
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
            'comfy_extras.nodes_minimax_h3': module('comfy_extras.nodes_minimax_h3',
                MiniMaxH3ImageToVideo=types.SimpleNamespace(execute=conditioning),
                MiniMaxH3ReferenceToVideo=types.SimpleNamespace(execute=reference_conditioning)),
            'comfy_extras.nodes_audio': module('comfy_extras.nodes_audio', VAEEncodeAudio=types.SimpleNamespace(execute=encode)),
            'nodes': module('nodes', VAEDecode=lambda: types.SimpleNamespace(decode=decode)),
        }
        self.patches = [patch.dict(sys.modules, mods), patch.object(ls, 'sample_single_stage', side_effect=sample),
                        patch.object(le, 'export_stitched_video', side_effect=export),
                        patch.object(ls, 'apply_motion_context', side_effect=context),
                        patch.object(ls, 'cleanup_segment_vram', side_effect=lambda **k: self.cleanups.append(k))]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def run_generation(self, samples=32000 * 27 + 7, **kwargs):
        self.audio = {'waveform': torch.ones(1, 1, samples), 'sample_rate': 32000}
        video_vae = kwargs.pop('video_vae', None)
        self.real_export = kwargs.pop('export_latent', False)
        self.result = ls.generate_lipsync(model=None, clip=None, video_vae=video_vae, audio_vae=None,
            audio=self.audio, first_frame=torch.ones(1, 32, 32, 3), prompt='speech', width=32, height=32,
            seed=2**64-1, **kwargs)
        return self.result[:5]

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
        self.assertEqual(len(self.cleanups), 2 * len(chunks) + 1)

    def test_sampling_allocations_released_before_decode(self):
        sample_impl = ls.sample_single_stage.side_effect
        allocations = []
        events = []

        def sample(**kwargs):
            result = sample_impl(**kwargs)
            allocations.extend(weakref.ref(tensor) for tensor in (
                kwargs['positive'][0][0],
                kwargs['latent']['noise_mask'].unbind()[0],
                result['samples'].unbind()[1],
            ))
            # The normal test recorder retains inputs; real sampling doesn't.
            self.samples.clear()
            events.append('sample')
            return result

        def cleanup(**kwargs):
            self.assertTrue(kwargs['enabled'])
            events.append('cleanup')

        def decode(vae, latent):
            self.assertEqual(events[-2:], ['sample', 'cleanup'])
            self.assertTrue(all(ref() is None for ref in allocations))
            self.assertEqual(latent['samples'].device.type, 'cpu')
            events.append('decode')
            return (torch.zeros(124, 2, 2, 3),)

        with patch.object(ls, 'sample_single_stage', new=sample), \
                patch.object(ls, 'cleanup_segment_vram', side_effect=cleanup), \
                patch.object(sys.modules['nodes'], 'VAEDecode', return_value=types.SimpleNamespace(decode=decode)):
            self.run_generation(samples=1)
        self.assertIn('decode', events)

    def test_model_cleanup_can_be_disabled(self):
        self.run_generation(clear_vram_between_chunks=False)
        self.assertTrue(self.cleanups)
        self.assertTrue(all(call['enabled'] is False for call in self.cleanups))

    def test_refinement_export_can_be_skipped_in_both_modes(self):
        for run in (self.run_generation, self.run_reference_generation):
            self.samples.clear()
            with patch.object(le, 'export_stitched_video', side_effect=AssertionError('export must be skipped')):
                images, audio, _, count, report = run(export_refinement=False)
            self.assertEqual(len(images), count)
            self.assertIs(audio, self.audio)
            self.assertEqual(self.result[5:], (None, [], []))
            self.assertEqual(json.loads(report)['latent_output'], 'disabled')

    def test_one_sample_audio(self):
        images, _, _, count, report = self.run_generation(samples=1)
        self.assertEqual(count, 1)
        self.assertEqual(images.shape[0], 1)
        self.assertEqual(len(self.contexts), 0)
        self.assertEqual(json.loads(report)['chunks'][0]['sample_frames'], 124)

    def test_legacy_reference_toggle(self):
        self.reference_each_chunk = False
        _, _, _, _, report = self.run_generation(reference_image_each_chunk=False)
        self.assertFalse(json.loads(report)['reference_image_each_chunk'])

    def test_corrected_context_matches_exported_tail(self):
        tails = []
        def encode(frames):
            tails.append(frames.clone())
            return torch.ones(1, 24, mc.steps_for_frames(len(frames)), 2, 2)
        images, audio, fps, count, report = self.run_generation(
            video_vae=types.SimpleNamespace(encode=encode), color_stabilization=0.35,
            detail_stabilization=0.35)
        chunks = json.loads(report)['chunks']
        self.assertEqual(len(tails), len(chunks) - 1)
        self.assertEqual(count, 649)
        self.assertIs(audio, self.audio)
        for tail, chunk in zip(tails, chunks):
            torch.testing.assert_close(tail, images[chunk['end_frame']-22:chunk['end_frame']])
        self.assertEqual(json.loads(report)['context_source'], 'corrected decoded tail')

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

    def run_reference_generation(self, **kwargs):
        ref_node = importlib.import_module('mmx_test_package.nodes.director_lipsync_ref').MiniMaxH3DirectorRefAudioLipSync
        self.audio = {'waveform': torch.linspace(-0.3, 0.4, 44100 * 12 + 7).reshape(1, 1, -1).repeat(1, 2, 1), 'sample_rate': 44100}
        self.original_waveform = self.audio['waveform'].clone()
        video_vae = kwargs.pop('video_vae', None)
        self.real_export = kwargs.pop('export_latent', False)
        self.result = ref_node().execute(model=None, clip=None, video_vae=video_vae, audio_vae=None, audio=self.audio,
            ref_image_1=torch.ones(1, 32, 32, 3), ref_image_2=torch.zeros(1, 32, 32, 3),
            prompt='The character in <Picture 1> in the room in <Picture 2>.', width=32, height=32, **kwargs)
        return self.result[:5]

    def test_whole_video_export_includes_all_corrected_frames_and_global_conditioning(self):
        encoded_frames = []
        def encode(frames):
            encoded_frames.append(frames.clone())
            return torch.full((1, 24, mc.steps_for_frames(len(frames)), 2, 2), 0.4)
        images, audio, _, count, report = self.run_reference_generation(
            video_vae=types.SimpleNamespace(encode=encode), export_latent=True,
            appearance_reference=torch.full((1, 32, 32, 3), 0.3),
            color_stabilization=0.35, detail_stabilization=0.35,
            chunk_seconds=5.17, chunk_prompts='["first", "second", "third"]')
        latent, positive, negative = self.result[5:]
        padded = ls.minimax_align_frame_count(count)
        self.assertEqual(count, 289)
        # First two encodes are corrected context tails. Export windows overlap
        # by five lookahead frames, which are not duplicated in the output.
        windows = encoded_frames[2:]
        self.assertTrue(all(len(window) <= 73 for window in windows))
        stitched = torch.cat([window[:-5] for window in windows[:-1]] + windows[-1:])
        self.assertEqual(len(stitched), padded)
        self.assertTrue(torch.equal(stitched[:count], images))
        self.assertTrue(torch.equal(stitched[count:], images[-1:].expand(padded-count, -1, -1, -1)))
        self.assertEqual(tuple(latent['samples'].shape), (1, 24, mc.steps_for_frames(padded), 2, 2))
        self.assertEqual(latent['minimax_h3_frame_count'], count)
        self.assertFalse(latent['samples'].requires_grad)
        self.assertEqual(negative, [])
        self.assertEqual(len(positive[0][1]['minimax_refs']), 2)
        self.assertNotIn('minimax_keyframes', positive[0][1])
        self.assertEqual(positive[0][1]['minimax_frame_count'], padded)
        self.assertEqual(self.reference_calls[-1]['prompt'], 'The character in <Picture 1> in the room in <Picture 2>.')
        self.assertEqual(json.loads(report)['latent_padding_frames'], padded-count)
        self.assertIs(audio, self.audio)
        self.assertTrue(torch.equal(audio['waveform'], self.original_waveform))

    def test_fl2va_whole_export_pads_very_short_audio_and_keeps_global_prompt(self):
        encoded_frames = []
        def encode(frames):
            encoded_frames.append(frames.clone())
            return torch.zeros(1, 24, mc.steps_for_frames(len(frames)), 2, 2)
        images, _, _, count, _ = self.run_generation(samples=1, export_latent=True,
            video_vae=types.SimpleNamespace(encode=encode), chunk_prompts='["chunk suffix"]')
        self.assertEqual(count, 1)
        self.assertEqual(len(encoded_frames[-1]), 5)
        self.assertTrue(torch.equal(encoded_frames[-1], images.expand(5, -1, -1, -1)))
        self.assertEqual(self.conditioning_calls[-1]['prompt'], 'speech')
        self.assertEqual(self.result[5]['samples'].shape[2], 2)
        self.assertEqual(self.result[6][0][1]['minimax_frame_count'], 5)
        self.assertNotIn('minimax_keyframes', self.result[6][0][1])

    def test_reference_images_persist_and_audio_is_bitwise_unchanged(self):
        images, audio, fps, count, report = self.run_reference_generation()
        self.assertIs(audio, self.audio)
        self.assertTrue(torch.equal(audio['waveform'], self.original_waveform))
        self.assertEqual(audio['sample_rate'], 44100)
        self.assertEqual(audio['waveform'].shape[1], 2)
        self.assertEqual(count, 289)
        self.assertEqual(len(images), count)
        self.assertEqual(len(self.reference_calls), 2)
        self.assertEqual(len(self.contexts), 1)
        for call, sample in zip(self.reference_calls, self.samples):
            self.assertEqual(list(call['ref_images']), ['ref_image_1', 'ref_image_2'])
            self.assertEqual(len(sample['positive'][0][1]['minimax_refs']), 2)
        parsed = json.loads(report)
        self.assertEqual(parsed['generation_mode'], 'ref2va')
        self.assertEqual(parsed['reference_image_count'], 2)
        self.assertTrue(parsed['source_audio_locked'])

    def test_reference_path_rejects_audio_regeneration(self):
        for setting in ({'audio_denoise': 0.1}, {'audio_noise_mask': torch.zeros(1, 1, 1)}):
            with self.assertRaisesRegex(ValueError, 'locks the input audio'):
                ls.generate_lipsync(model=None, clip=None, video_vae=None, audio_vae=None,
                    audio={'waveform': torch.zeros(1, 1, 32000), 'sample_rate': 32000},
                    prompt='test', width=32, height=32, generation_mode='ref2va',
                    ref_images={'ref_image_1': torch.zeros(1, 32, 32, 3)}, **setting)
        self.assertEqual(len(self.samples), 0)

    def test_reference_stabilization_requires_full_scene_reference(self):
        with self.assertRaisesRegex(ValueError, 'appearance_reference'):
            self.run_reference_generation(color_stabilization=0.35)
        self.assertEqual(len(self.reference_calls), 0)

    def test_anchor_requires_full_scene_reference_for_ref2va(self):
        with self.assertRaisesRegex(ValueError, 'full-scene appearance_reference'):
            self.run_reference_generation(first_frame_anchor_strength=1)
        self.assertEqual(len(self.samples), 0)

    def test_reference_anchor_is_fixed_across_later_chunks_with_audio_unchanged(self):
        def encode(frames):
            t = 1 if len(frames) == 1 else mc.steps_for_frames(len(frames))
            return torch.full((1, 24, t, 2, 2), float(frames.mean()))
        _, audio, _, count, report = self.run_reference_generation(
            video_vae=types.SimpleNamespace(encode=encode), chunk_seconds=5.17,
            appearance_reference=torch.full((1, 32, 32, 3), 0.7), first_frame_anchor_strength=1)
        self.assertEqual(len(self.samples), 3)
        self.assertEqual(count, 289)
        self.assertTrue(torch.equal(audio['waveform'], self.original_waveform))
        for sample in self.samples[1:]:
            metadata = sample['positive'][0][1]
            keys = metadata['minimax_keyframes']
            torch.testing.assert_close(keys[0]['latent'], torch.full_like(keys[0]['latent'], 0.7))
            self.assertTrue(all(k['latent'].count_nonzero() == 0 for k in keys[1:]))
            self.assertEqual(len(metadata['minimax_refs']), 2)
        self.assertEqual([c['first_frame_anchor_applied'] for c in json.loads(report)['chunks']], [False, True, True])

    def test_reference_correction_feeds_exported_tail_without_changing_audio(self):
        tails, encoded_tails = [], []
        def encode(frames):
            tails.append(frames.clone())
            latent = torch.full((1, 24, mc.steps_for_frames(len(frames)), 2, 2), float(len(tails)))
            encoded_tails.append(latent.clone())
            return latent

        # A nonzero decode differs from the reference; use real correction,
        # then check the actual context passed into the following sample.
        reference = torch.full((1, 32, 32, 3), 0.3)
        original_reference = reference.clone()
        images, audio, fps, count, report = self.run_reference_generation(
            video_vae=types.SimpleNamespace(encode=encode), chunk_seconds=5.17,
            appearance_reference=reference, color_stabilization=0.35, detail_stabilization=0.35)
        parsed = json.loads(report)
        self.assertEqual(len(tails), 2)
        self.assertEqual(len(self.reference_calls), 3)
        self.assertEqual(count, 289)
        self.assertEqual(fps, 24)
        self.assertEqual(parsed['context_source'], 'corrected decoded tail')
        self.assertTrue(parsed['source_audio_locked'])
        self.assertIs(audio, self.audio)
        self.assertTrue(torch.equal(audio['waveform'], self.original_waveform))
        self.assertTrue(torch.equal(reference, original_reference))
        for tail, encoded, context, chunk in zip(tails, encoded_tails, self.contexts, parsed['chunks']):
            self.assertTrue(torch.equal(tail, images[chunk['end_frame']-22:chunk['end_frame']]))
            self.assertTrue(torch.equal(context['context_latent']['samples'].unbind()[0], encoded))
            self.assertLessEqual(tail.max().item(), 1.0)
        for sample in self.samples:
            self.assertEqual(len(sample['positive'][0][1]['minimax_refs']), 2)


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
            gen = next(n for n in nodes.values() if n['type'] in ('MiniMaxH3DirectorLongAudioLipSync', 'MiniMaxH3DirectorRefAudioLipSync'))
            create = next(n for n in nodes.values() if n['type'] in ('CreateVideo', 'MiniMaxH3SaveVideoExactAudio'))
            audio_link = next(i['link'] for i in create['inputs'] if i['name'] == 'audio')
            self.assertEqual(links[audio_link][1:3], [gen['id'], 1])
            values = gen['widgets_values']
            self.assertEqual([(o['name'], o['type']) for o in gen['outputs'][5:]],
                             [('latent', 'LATENT'), ('positive', 'CONDITIONING'), ('negative', 'CONDITIONING')])
            self.assertEqual(values[3:5], [10.0, 22])
            self.assertEqual(values[6], 'fixed')
            if gen['type'] == 'MiniMaxH3DirectorRefAudioLipSync':
                self.assertEqual(create['type'], 'MiniMaxH3SaveVideoExactAudio')
                stabilized = path.stem.endswith('_stabilized')
                strength = 0.35 if stabilized else 0.0
                self.assertEqual(values[12:], [True, '', strength, strength, 'match', 0.0, True])
                appearance_link = next(i['link'] for i in gen['inputs'] if i['name'] == 'appearance_reference')
                if stabilized:
                    self.assertIsNotNone(appearance_link)
                    source = nodes[links[appearance_link][1]]
                    self.assertEqual(source['type'], 'LoadImage')
                    self.assertEqual(source['widgets_values'][0], 'appearance_reference.png')
            else:
                self.assertEqual(values[12], 0.0)  # lock source audio
                self.assertEqual(values[14], '')  # optional chunk prompts
                self.assertEqual(values[15:], [True, 0.0, 0.0, 0.0, True])


if __name__ == '__main__':
    unittest.main()
