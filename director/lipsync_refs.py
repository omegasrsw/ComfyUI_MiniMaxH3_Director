"""Ordered image references for the source-audio-locked ref2va path."""

import torch


def validate_reference_image(image, name):
    if (not isinstance(image, torch.Tensor) or image.ndim != 4
            or image.shape[0] != 1 or min(image.shape[1:3]) < 1 or image.shape[-1] not in (3, 4)
            or not image.is_floating_point() or not torch.isfinite(image).all().item()):
        raise ValueError(f"{name} must contain exactly one finite RGB/RGBA IMAGE.")


def normalize_lipsync_refs(ref_images):
    if not isinstance(ref_images, dict) or not ref_images:
        raise ValueError("ref2va lip sync requires ref_image_1 (character/reference image).")
    allowed = [f"ref_image_{i}" for i in range(1, 10)]
    if any(k not in allowed for k in ref_images):
        raise ValueError("Reference image slots must be ref_image_1 through ref_image_9.")
    refs = {k: ref_images[k] for k in allowed if ref_images.get(k) is not None}
    if list(refs) != allowed[:len(refs)] or not refs:
        # The official node numbers nonempty refs in insertion order. Reject
        # holes instead of silently calling slot 3 <Picture 2> in the prompt.
        raise ValueError("Connect consecutive reference slots starting at ref_image_1; do not leave gaps.")
    for name, image in refs.items():
        validate_reference_image(image, name)
    return refs
