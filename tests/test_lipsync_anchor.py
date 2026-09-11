import importlib
import unittest
import torch
from test_lipsync import ROOT

anchor = importlib.import_module('mmx_test_package.director.lipsync_anchor')


class AnchorTests(unittest.TestCase):
    def test_original_anchor_replaces_only_oldest_hidden_keyframe(self):
        original = torch.full((1, 24, 1, 2, 2), 0.75)
        keys = [{'director_context_index': i, 'resolved_frame_index': 0,
                 'latent': torch.full_like(original, i / 10)} for i in (0, 1, 5, 9)]
        refs = [{'kind': 'image', 'latent': torch.ones_like(original)}]
        positive = [[torch.ones(1, 7, 8), {'minimax_keyframes': keys, 'minimax_refs': refs, 'minimax_frame_count': 124}]]
        out = anchor.anchor_context_head(positive, original, 1)
        self.assertTrue(torch.equal(out[0][1]['minimax_keyframes'][0]['latent'], original))
        self.assertEqual(keys[0]['latent'].count_nonzero(), 0)
        self.assertIs(out[0][1]['minimax_refs'], refs)
        self.assertEqual(out[0][1]['minimax_frame_count'], 124)
        self.assertIs(out[0][0], positive[0][0])
        for old, new in zip(keys[1:], out[0][1]['minimax_keyframes'][1:]):
            self.assertIs(new, old)
        blended = anchor.anchor_context_head(positive, original, 0.5)
        torch.testing.assert_close(blended[0][1]['minimax_keyframes'][0]['latent'], original * 0.5)
        self.assertIs(anchor.anchor_context_head(positive, None, 0), positive)

    def test_invalid_settings_or_missing_context_fail(self):
        for value in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                anchor.validate_anchor_strength(value)
        with self.assertRaisesRegex(RuntimeError, 'frame-zero'):
            anchor.anchor_context_head([[torch.ones(1), {}]], torch.zeros(1, 24, 1, 2, 2), 1)
