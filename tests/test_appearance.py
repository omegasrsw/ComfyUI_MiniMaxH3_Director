import unittest
import torch

from director.appearance import AppearanceStabilizer, _smooth


class AppearanceTests(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(41)
        self.reference = 0.3 + 0.2 * torch.rand(1, 16, 16, 3, generator=gen)
        self.frames = (self.reference.repeat(43, 1, 1, 1) * 1.1 + torch.tensor([0.06, 0.02, -0.03])).clamp(0, 1)

    def test_disabled_is_exact_noop(self):
        filt = AppearanceStabilizer(self.reference, 16, 16)
        self.assertIs(filt.process(self.frames), self.frames)

    def test_chunk_boundaries_do_not_reset_filter(self):
        a = AppearanceStabilizer(self.reference, 16, 16, color_strength=0.5, detail_strength=0.5)
        b = AppearanceStabilizer(self.reference, 16, 16, color_strength=0.5, detail_strength=0.5)
        full = a.process(self.frames)
        split = torch.cat([b.process(self.frames[:13]), b.process(self.frames[13:34]), b.process(self.frames[34:])])
        torch.testing.assert_close(full, split, rtol=0, atol=1e-6)

    def test_color_moves_toward_reference_without_mutating_source(self):
        saved = self.frames.clone()
        filt = AppearanceStabilizer(self.reference, 16, 16, color_strength=1)
        result = filt.process(self.frames)
        expected = self.reference.mean(dim=(0, 1, 2))
        before = (self.frames[-1].mean(dim=(0, 1)) - expected).abs().mean()
        after = (result[-1].mean(dim=(0, 1)) - expected).abs().mean()
        self.assertLess(after, before * 0.7)
        torch.testing.assert_close(self.frames, saved)
        self.assertTrue(((result >= 0) & (result <= 1)).all())

    def test_detail_filter_only_reduces_excess(self):
        ref = torch.full((1, 16, 16, 3), 0.5)
        frames = ref.repeat(43, 1, 1, 1)
        frames[:, ::2, ::2] += 0.15
        frames[:, 1::2, 1::2] -= 0.15
        filt = AppearanceStabilizer(ref, 16, 16, detail_strength=1)
        result = filt.process(frames)
        def detail(x):
            x = x[-1:].movedim(-1, 1)
            return (x - _smooth(x)).square().mean()
        self.assertLess(detail(result), detail(frames) * 0.5)
        flat = AppearanceStabilizer(ref, 16, 16, detail_strength=1).process(ref)
        torch.testing.assert_close(flat, ref)

    def test_invalid_strength(self):
        for value in (-0.01, 1.01, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                AppearanceStabilizer(self.reference, 16, 16, color_strength=value)


if __name__ == '__main__':
    unittest.main()
