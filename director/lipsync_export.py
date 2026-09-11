"""Whole-timeline refinement outputs, rebuilt after pixel-space stitching."""

import logging
import time
import torch

from .core_sampling import _unpack_node_output
from .h3_motion_context import steps_for_frames

log = logging.getLogger(__name__)


def encode_stitched_frames(frames, video_vae, width, height):
    """Bound VAE input conversions while preserving H3's global 17-frame phase.

    H3 encodes independent 17-frame clips into five tokens, then drops three
    tokens once at EOF. Each interior call includes a five-frame lookahead
    clip so its local EOF drop only affects tokens we discard. Keep 20 tokens
    for each 68-frame interior window; the final window keeps its EOF tokens.
    These windows use the stitched pixels, not the generation chunk latents.
    """
    from comfy.model_management import throw_exception_if_processing_interrupted

    total = len(frames)
    if total < 5 or (total - 5) % 17:
        raise ValueError("Stitched export requires the padded H3 17k+5 frame grid.")
    core_frames = 68
    windows = max(1, (total - 5 + core_frames - 1) // core_frames)
    output = None
    position = 0
    for index in range(windows):
        throw_exception_if_processing_interrupted()
        start = index * core_frames
        stop = min(total, start + core_frames + 5)
        log.info("H3 lip sync: whole-video latent export window %d/%d starting (%d frames)",
                 index + 1, windows, stop - start)
        started = time.perf_counter()
        encoded = video_vae.encode(frames[start:stop])
        expected = (1, 24, steps_for_frames(stop - start), height // 16, width // 16)
        if tuple(encoded.shape) != expected:
            raise RuntimeError(f"Stitched H3 video VAE returned {tuple(encoded.shape)}; expected {expected}.")
        keep = encoded.shape[2] if index == windows - 1 else core_frames // 17 * 5
        if output is None:
            output = torch.empty((1, 24, steps_for_frames(total), height // 16, width // 16),
                                 dtype=encoded.dtype, device="cpu")
        output[:, :, position:position + keep].copy_(encoded[:, :, :keep].detach())
        position += keep
        del encoded
        log.info("H3 lip sync: whole-video latent export window %d/%d complete in %.1fs (%d/%d tokens)",
                 index + 1, windows, time.perf_counter() - started, position, output.shape[2])
    if position != output.shape[2]:
        raise RuntimeError("Stitched H3 export did not cover the complete latent timeline.")
    return output


def cpu_snapshot(value):
    """Detach conditioning from model/device lifetime without mutating it."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def export_stitched_video(*, frames, frame_count, video_vae, clip, prompt,
                          width, height, generation_mode, first_frame=None,
                          ref_images=None, ref_image_size="match"):
    """Encode one padded, stitched timeline; never concatenate chunk latents.

    ``frames`` includes only terminal repeat padding on the H3 frame grid.
    Bound input conversion memory above the VAE's internal temporal/spatial tiles.
    """
    encoded = encode_stitched_frames(frames, video_vae, width, height)
    latent = {
        "samples": encoded.detach().cpu(),
        "minimax_h3_frame_count": int(frame_count),
        "minimax_h3_encoded_frame_count": len(frames),
        "minimax_h3_fps": 24.0,
    }
    # Do not retain a GPU encoder result while loading the text encoder.
    del encoded
    # These image-only conditioners do not use length to encode their images
    # or prompt. A 5-frame placeholder avoids allocating an unused full AV
    # timeline. Set the true full frame count explicitly below.
    log.info("H3 lip sync: building full-video refinement conditioning")
    if generation_mode == "ref2va":
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        positive, _ = _unpack_node_output(MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=video_vae, prompt=prompt, width=width, height=height,
            length=5, ref_image_size=ref_image_size, ref_images=ref_images))
    else:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        positive, _ = _unpack_node_output(MiniMaxH3ImageToVideo.execute(
            clip=clip, vae=video_vae, prompt=prompt, width=width, height=height,
            length=5, first_frame=first_frame))
    positive = cpu_snapshot(positive)
    for _, metadata in positive:
        # Stock H3 keyframes share the target spatial grid. An original-size
        # anchor would mismatch the upscaled target's token count. The Qwen
        # image embeddings remain; ref2va reference blocks have their own grid.
        metadata.pop("minimax_keyframes", None)
        metadata["minimax_frame_count"] = len(frames)
    # The generation path uses BasicGuider / CFG 1 with no negative prompt.
    # An empty CONDITIONING is the faithful negative output, not invented text.
    return latent, positive, []
