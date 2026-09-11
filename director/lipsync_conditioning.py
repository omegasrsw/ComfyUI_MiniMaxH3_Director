"""Global guidance for optional external refinement of corrected video frames."""

import logging
import torch

from .core_sampling import _unpack_node_output
from .frame_align import minimax_align_frame_count

log = logging.getLogger(__name__)


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


def build_refinement_conditioning(*, frame_count, video_vae, clip, prompt,
                                  width, height, generation_mode, first_frame=None,
                                  ref_images=None, ref_image_size="match"):
    """Encode global prompt/references only; never receive or encode video output.

    The official conditioners require an AV placeholder. Use length=5 to keep
    that discarded allocation small, then set guidance to the whole H3 grid.
    External video encoding must pad to this grid and trim after decoding.
    """
    log.info("H3 lip sync: building full-video positive/negative conditioning (prompt and references only)")
    if generation_mode == "ref2va":
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        positive, placeholder = _unpack_node_output(MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=video_vae, prompt=prompt, width=width, height=height,
            length=5, ref_image_size=ref_image_size, ref_images=ref_images))
    else:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        positive, placeholder = _unpack_node_output(MiniMaxH3ImageToVideo.execute(
            clip=clip, vae=video_vae, prompt=prompt, width=width, height=height,
            length=5, first_frame=first_frame))
    del placeholder
    positive = cpu_snapshot(positive)
    for _, metadata in positive:
        # FL2VA anchors use the original spatial grid and cannot be reused
        # after upscaling. Vision embeddings stay; Ref2VA blocks own their grid.
        metadata.pop("minimax_keyframes", None)
        metadata["minimax_frame_count"] = minimax_align_frame_count(frame_count)
    # Generation uses CFG 1 / BasicGuider with no negative prompt.
    return positive, []
