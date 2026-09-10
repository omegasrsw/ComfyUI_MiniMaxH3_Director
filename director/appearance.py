"""Conservative, optional appearance correction for continuous talking shots.

Reference statistics remain fixed for the entire recording. Correction controls
are smoothed on the output timeline, including across chunk boundaries. These
filters adjust color and excess fine detail; they do not restore identity/pose.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _smooth(image):
    return F.avg_pool2d(F.pad(image, (1, 1, 1, 1), mode="replicate"), 3, stride=1)


def _color_stats(image):
    small = F.interpolate(image, size=(64, 64), mode="area")
    return small.mean(dim=(2, 3)), small.std(dim=(2, 3), correction=0).clamp_min(0.02)


class AppearanceStabilizer:
    """Process NHWC frames in place-independent batches with constant history.

    Color correction moves RGB mean/std toward the reference, with bounded
    gains/offsets. Detail correction can only attenuate the 3x3 high-pass band,
    never sharpen it. Neither operation blends reference pixels into the mouth.
    """

    def __init__(self, reference, width, height, *, color_strength=0.0,
                 detail_strength=0.0, fps=24.0):
        for name, value in (("color_stabilization", color_strength), ("detail_stabilization", detail_strength)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        self.color_strength = float(color_strength)
        self.detail_strength = float(detail_strength)
        self.enabled = bool(color_strength or detail_strength)
        if not self.enabled:
            return
        ref = reference[:1, ..., :3].detach().cpu().float().movedim(-1, 1)
        ref = F.interpolate(ref, (height, width), mode="bilinear", align_corners=False, antialias=True)
        if not torch.isfinite(ref).all().item():
            raise ValueError("The appearance reference must contain finite pixels.")
        self.mean, self.std = _color_stats(ref)
        self.detail = (ref - _smooth(ref)).square().mean().sqrt().clamp_min(1e-4)
        self.gain = torch.ones(3)
        self.offset = torch.zeros(3)
        self.detail_gain = torch.tensor(1.0)
        # Half-second smoothing carries through every chunk; no per-chunk reset.
        self.decay = math.exp(-1.0 / (float(fps) * 0.5))

    def process(self, frames):
        if not self.enabled:
            return frames
        result = torch.empty_like(frames, device="cpu", dtype=torch.float32)
        for start in range(0, len(frames), 8):
            x = frames[start:start + 8].detach().cpu().float().movedim(-1, 1)
            means, stds = _color_stats(x)
            gains = (self.std / stds).clamp(0.85, 1.15)
            offsets = (self.mean - means * gains).clamp(-0.08, 0.08)
            gains = 1 + self.color_strength * (gains - 1)
            offsets = self.color_strength * offsets
            low = _smooth(x) if self.detail_strength else None
            if low is not None:
                energy = (x - low).square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-4)
                targets = 1 + self.detail_strength * ((self.detail / energy).clamp(0.5, 1.0) - 1)
            for i in range(len(x)):
                self.gain = self.decay * self.gain + (1 - self.decay) * gains[i]
                self.offset = self.decay * self.offset + (1 - self.decay) * offsets[i]
                frame = x[i]
                if low is not None:
                    self.detail_gain = self.decay * self.detail_gain + (1 - self.decay) * targets[i]
                    frame = low[i] + self.detail_gain * (frame - low[i])
                frame = frame * self.gain[:, None, None] + self.offset[:, None, None]
                result[start + i] = frame.clamp(0, 1).movedim(0, -1)
        return result
