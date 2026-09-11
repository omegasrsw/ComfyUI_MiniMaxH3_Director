"""Experimental fixed-image anchor inside the discarded context head."""

import math
import torch
import torch.nn.functional as F

from .h3_context_patches import CTX_FRAME_KEY


def validate_anchor_strength(strength):
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("first_frame_anchor_strength must be between 0 and 1.")


def encode_scene_anchor(image, vae, width, height):
    """Ref2VA uses an explicitly supplied full scene, never a character crop."""
    pixels = image[:1, ..., :3].movedim(-1, 1)
    pixels = F.interpolate(pixels, (height, width), mode="bilinear", align_corners=False, antialias=True)
    return vae.encode(pixels.movedim(1, -1)).detach().cpu().clone()


def original_fl2va_anchor(positive):
    for _, metadata in positive:
        for keyframe in metadata.get("minimax_keyframes", []):
            if keyframe.get("resolved_frame_index") == 0 and CTX_FRAME_KEY not in keyframe:
                return keyframe["latent"].detach().cpu().clone()
    raise RuntimeError("FL2VA did not provide the original first-frame latent anchor.")


def anchor_context_head(positive, anchor, strength):
    """Replace/blend only the oldest hidden keyframe, preserving recent motion.

    This is a latent-image interpolation, not CFG/attention weighting. At 1
    the first hidden context keyframe is the exact original image encoding.
    All remaining context blocks, timestamps and reference blocks are retained.
    """
    validate_anchor_strength(strength)
    if strength == 0:
        return positive
    if (not isinstance(anchor, torch.Tensor) or anchor.ndim != 5
            or anchor.shape[:3] != (1, 24, 1)):
        raise ValueError("The appearance anchor must be one H3 image latent [1,24,1,H,W].")
    result = []
    for embedding, metadata in positive:
        keyframes = metadata.get("minimax_keyframes", [])
        matches = [i for i, kf in enumerate(keyframes) if kf.get(CTX_FRAME_KEY) == 0]
        if len(matches) != 1 or len(keyframes) < 2:
            raise RuntimeError("Appearance anchoring needs one frame-zero context block plus recent motion context.")
        index = matches[0]
        old = keyframes[index]["latent"]
        if old.shape != anchor.shape:
            raise RuntimeError("Original image and motion context have incompatible latent shapes.")
        replacement = anchor.to(old)
        replacement = replacement.clone() if strength == 1 else torch.lerp(old, replacement, strength)
        updated = list(keyframes)
        updated[index] = {**keyframes[index], "latent": replacement}
        result.append([embedding, {**metadata, "minimax_keyframes": updated}])
    return result
