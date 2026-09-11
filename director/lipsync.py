"""Audio-clocked H3 generation. Only the final chunk may crop grid overshoot.

All boundaries are integer video frames on one 24 fps timeline. Full chunks
end on H3's 17k+5 grid, so their latent tails can be reused without dropping
any previously exported frames. Audio is sliced at absolute sample offsets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import logging
import time

import torch
import torch.nn.functional as F

from .frame_align import minimax_align_frame_count
from .h3_motion_context import CONTEXT_FRAME_CHOICES, apply_motion_context
from .core_sampling import _unpack_node_output, sample_single_stage
from .vram_cleanup import cleanup_segment_vram
from .appearance import AppearanceStabilizer
from .lipsync_refs import normalize_lipsync_refs, validate_reference_image
from .lipsync_anchor import validate_anchor_strength, encode_scene_anchor, original_fl2va_anchor, anchor_context_head

FPS = 24
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start_frame: int
    end_frame: int
    context_frames: int
    sample_frames: int

    @property
    def audio_start_frame(self):
        return self.start_frame - self.context_frames

    @property
    def visible_frames(self):
        return self.end_frame - self.start_frame


def plan_audio_chunks(sample_count, sample_rate, chunk_seconds=10.0, context_frames=22):
    """Bound the *whole sample*, including context, by chunk_seconds <= 15.

    Final samples are padded to at least 124 frames (H3's training range);
    only real audio duration, rounded up to one video frame, is exported.
    """
    if sample_count <= 0 or sample_rate <= 0:
        raise ValueError("Lip sync requires non-empty audio and a positive sample rate.")
    if not math.isfinite(chunk_seconds) or not 124 / FPS <= chunk_seconds <= 15:
        raise ValueError("chunk_seconds must be between 5.17 and 15 seconds.")
    if context_frames not in CONTEXT_FRAME_CHOICES:
        raise ValueError(f"context_frames must be one of {CONTEXT_FRAME_CHOICES}.")
    max_frames = 5 + ((math.floor(chunk_seconds * FPS) - 5) // 17) * 17
    total_frames = (int(sample_count) * FPS + int(sample_rate) - 1) // int(sample_rate)
    chunks, start = [], 0
    while start < total_frames:
        context = int(context_frames) if chunks else 0
        visible = min(max_frames - context, total_frames - start)
        sample_frames = max(124, minimax_align_frame_count(visible + context))
        chunks.append(AudioChunk(len(chunks), start, start + visible, context, sample_frames))
        start += visible
    return chunks


def sample_offset(frame, sample_rate):
    # Integer round-half-up; use absolute boundaries instead of summing rounded durations.
    return (int(frame) * int(sample_rate) + FPS // 2) // FPS


def validate_audio(audio):
    if not isinstance(audio, dict):
        raise ValueError("Connect a ComfyUI AUDIO input.")
    waveform = audio.get("waveform")
    rate = audio.get("sample_rate")
    if (not isinstance(waveform, torch.Tensor) or waveform.ndim != 3
            or waveform.shape[0] != 1 or waveform.shape[1] not in (1, 2)
            or waveform.shape[-1] == 0):
        raise ValueError("Audio must be one non-empty mono/stereo waveform [1, channels, samples].")
    if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
        raise ValueError("Audio sample_rate must be a positive integer.")
    if not waveform.is_floating_point() or not torch.isfinite(waveform).all().item():
        raise ValueError("Audio waveform must contain finite floating-point samples.")
    return waveform, rate


def slice_chunk_audio(audio, chunk):
    """Include the source soundtrack under the context head, pad only at EOF."""
    waveform, rate = audio["waveform"], audio["sample_rate"]
    start = sample_offset(chunk.audio_start_frame, rate)
    end = sample_offset(chunk.audio_start_frame + chunk.sample_frames, rate)
    piece = waveform[..., start:min(end, waveform.shape[-1])].detach().cpu().float()
    piece = F.pad(piece, (0, end - start - piece.shape[-1]))
    # H3's audio VAE represents stereo, including when the source is mono.
    if piece.shape[1] == 1:
        piece = piece.repeat(1, 2, 1)
    return {"waveform": piece, "sample_rate": rate}


def audio_mask_for_chunk(mask, chunk, audio_samples, total_frames, strength):
    """Convert a per-video-frame MASK to an audio-time envelope, if connected.

    Spatial masks (e.g. the supplied workflow's amplitude circles) have no
    spatial interpretation in audio. Their maximum gives one strength per
    frame. Map that envelope to the 40 Hz audio grid without stretching time.
    """
    if mask is None:
        return torch.full_like(audio_samples, strength)
    if mask.ndim != 3 or mask.shape[0] not in (1, total_frames):
        raise ValueError("audio_noise_mask must have 1 frame or ceil(audio seconds * 24) frames.")
    envelope = mask.detach().to(device=audio_samples.device, dtype=audio_samples.dtype).amax(dim=(1, 2))
    if not torch.isfinite(envelope).all().item() or envelope.min() < 0 or envelope.max() > 1:
        raise ValueError("audio_noise_mask values must be finite and between 0 and 1.")
    times = chunk.audio_start_frame + torch.arange(audio_samples.shape[-1], device=envelope.device) * (FPS / 40)
    indices = times.floor().long().clamp(0, envelope.numel() - 1)
    values = envelope[indices]
    return values.reshape(1, 1, 1, -1).expand_as(audio_samples).contiguous()


def inject_source_audio(latent, audio_vae, audio, chunk, *, mask=None, total_frames, strength=0.0):
    """Replace the empty audio stream and explicitly set its denoise mask.

    Zero preserves the source audio latent during sampling. A nonzero mask
    allows experimentation like the user's SetLatentNoiseMask branch.
    """
    from comfy.nested_tensor import NestedTensor
    from comfy_extras.nodes_audio import VAEEncodeAudio

    video, empty_audio = latent["samples"].unbind()
    if video.ndim != 5 or empty_audio.ndim != 4 or empty_audio.shape[1:3] != (32, 2):
        raise ValueError("Expected MiniMax H3 video and stereo audio latents; check the connected VAEs.")
    encoded = _unpack_node_output(VAEEncodeAudio.execute(audio_vae, slice_chunk_audio(audio, chunk)))[0]["samples"]
    if encoded.ndim != 4 or encoded.shape[:-1] != empty_audio.shape[:-1]:
        raise ValueError("audio_vae must be the MiniMax H3 stereo audio VAE.")
    target = empty_audio.shape[-1]
    # VAE length rounding may differ by one token from H3 temporal_shape().
    # Do not silently accept an incompatible encoder or a truncated waveform.
    if abs(encoded.shape[-1] - target) > 1:
        raise ValueError(f"H3 audio grid mismatch: encoded {encoded.shape[-1]} steps, expected {target}.")
    encoded = encoded[..., :target].to(empty_audio)
    if encoded.shape[-1] < target:
        encoded = torch.cat((encoded, encoded[..., -1:].expand(*encoded.shape[:-1], target - encoded.shape[-1])), dim=-1)
    audio_mask = audio_mask_for_chunk(mask, chunk, encoded, total_frames, strength)
    result = dict(latent)
    result["samples"] = NestedTensor((video, encoded))
    result["noise_mask"] = NestedTensor((torch.ones_like(video), audio_mask))
    return result


def parse_chunk_prompts(raw, count, fallback):
    if not str(raw or "").strip():
        return [fallback] * count
    try:
        prompts = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("chunk_prompts must be a JSON array of strings.") from exc
    if not isinstance(prompts, list) or len(prompts) != count or any(not isinstance(p, str) for p in prompts):
        raise ValueError(f"chunk_prompts must contain exactly {count} strings, one per generated chunk.")
    return [f"{fallback}\n{p}".strip() for p in prompts]


def generate_lipsync(*, model, clip, video_vae, audio_vae, audio, first_frame=None,
                     prompt, width, height, chunk_seconds=10.0, context_frames=22,
                     seed=0, steps=25, sampler="res_multistep", scheduler="simple",
                     shift_video=12.0, shift_audio=3.0, audio_denoise=0.0,
                     clear_vram_between_chunks=True, sigmas=None, audio_noise_mask=None,
                     chunk_prompts="", reference_image_each_chunk=True,
                     color_stabilization=0.0, detail_stabilization=0.0,
                     generation_mode="fl2va", ref_images=None, ref_image_size="match",
                     appearance_reference=None, first_frame_anchor_strength=0.0):
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
    from comfy.utils import ProgressBar
    import comfy.model_management as mm
    from nodes import VAEDecode

    waveform, rate = validate_audio(audio)
    validate_anchor_strength(first_frame_anchor_strength)
    if width < 32 or height < 32 or width % 32 or height % 32:
        raise ValueError("width and height must be positive multiples of 32.")
    if not math.isfinite(audio_denoise) or not 0 <= audio_denoise <= 1:
        raise ValueError("audio_denoise must be between 0 and 1.")
    if generation_mode not in ("fl2va", "ref2va"):
        raise ValueError("generation_mode must be fl2va or ref2va.")
    if generation_mode == "ref2va":
        ref_images = normalize_lipsync_refs(ref_images)
        if audio_denoise != 0 or audio_noise_mask is not None:
            raise ValueError("ref2va lip sync locks the input audio: audio_denoise must be 0 and audio_noise_mask disconnected.")
        if ref_image_size not in ("match", "max"):
            raise ValueError("ref_image_size must be match or max.")
        if first_frame is not None:
            raise ValueError("ref2va uses reference images, not a first-frame anchor.")
        if appearance_reference is not None:
            validate_reference_image(appearance_reference, "appearance_reference")
        if (color_stabilization or detail_stabilization) and appearance_reference is None:
            raise ValueError("Connect appearance_reference to enable ref2va color/detail stabilization; use an image of the intended full scene.")
        if first_frame_anchor_strength and appearance_reference is None:
            raise ValueError("Connect a full-scene appearance_reference for ref2va first_frame_anchor_strength.")
        appearance_image = appearance_reference
    else:
        validate_reference_image(first_frame, "first_frame")
        if ref_images is not None or appearance_reference is not None:
            raise ValueError("Use the ref2va lip-sync node for multiple reference images.")
        appearance_image = first_frame
    chunks = plan_audio_chunks(waveform.shape[-1], rate, chunk_seconds, context_frames)
    total_frames = chunks[-1].end_frame
    prompts = parse_chunk_prompts(chunk_prompts, len(chunks), prompt)
    appearance = AppearanceStabilizer(appearance_image, width, height,
                                     color_strength=color_stabilization,
                                     detail_strength=detail_stabilization)
    if audio_noise_mask is not None:
        # Validate before loading the expensive text encoder / diffusion model.
        audio_mask_for_chunk(audio_noise_mask, chunks[0], torch.zeros(1, 32, 2, 1), total_frames, audio_denoise)
    refinement_frame_count = minimax_align_frame_count(total_frames)
    progress = ProgressBar(len(chunks) + 1)
    log.info("H3 lip sync: %d frames, %d generation chunks; corrected images and global conditioning outputs",
             total_frames, len(chunks))
    output = None
    full_positive, full_negative = [], []
    previous = None
    fixed_anchor = None
    records = []
    try:
        with torch.inference_mode():
            for chunk, chunk_prompt in zip(chunks, prompts):
                mm.throw_exception_if_processing_interrupted()
                log.info("H3 lip sync: chunk %d/%d conditioning", chunk.index + 1, len(chunks))
                if generation_mode == "ref2va":
                    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
                    positive, latent = _unpack_node_output(MiniMaxH3ReferenceToVideo.execute(
                        clip=clip, vae=video_vae, prompt=chunk_prompt, width=width, height=height,
                        length=chunk.sample_frames, ref_image_size=ref_image_size, ref_images=ref_images,
                    ))
                    if first_frame_anchor_strength and fixed_anchor is None:
                        fixed_anchor = encode_scene_anchor(appearance_reference, video_vae, width, height)
                else:
                    positive, latent = _unpack_node_output(MiniMaxH3ImageToVideo.execute(
                        clip=clip, vae=video_vae, prompt=chunk_prompt, width=width, height=height,
                        # Keep the portrait's vision embeddings; motion context
                        # replaces only its frame-0 latent anchor.
                        length=chunk.sample_frames,
                        first_frame=first_frame if previous is None or reference_image_each_chunk else None,
                    ))
                    if first_frame_anchor_strength and fixed_anchor is None:
                        fixed_anchor = original_fl2va_anchor(positive)
                if previous is not None:
                    positive, trim, previous_trim = apply_motion_context(
                        positive, latent, vae=video_vae, context_length=chunk.context_frames,
                        context_latent=previous, continue_audio=False,
                    )
                    if trim != chunk.context_frames or previous_trim:
                        raise RuntimeError("Lip sync context alignment changed; refusing a shifted audio/video join.")
                    positive = anchor_context_head(positive, fixed_anchor, first_frame_anchor_strength)
                latent = inject_source_audio(latent, audio_vae, audio, chunk, mask=audio_noise_mask,
                                             total_frames=total_frames, strength=audio_denoise)
                chunk_seed = (int(seed) + chunk.index) % (2**64)
                log.info("H3 lip sync: chunk %d/%d sampling", chunk.index + 1, len(chunks))
                sampled = sample_single_stage(
                    model=model, positive=positive, negative=[], latent=latent, seed=chunk_seed,
                    cfg=1.0, steps=steps, sampler_name=sampler, scheduler=scheduler,
                    shift_video=shift_video, shift_audio=shift_audio, sigmas=sigmas,
                )
                # Decoding only needs video. Release the AV sampling inputs,
                # masks and conditioning before asking ComfyUI to load the VAE.
                # Keep the sampled video on CPU so model unloading can reclaim
                # VRAM even when the sampler's intermediate device is CUDA.
                video = sampled["samples"].unbind()[0].detach().cpu()
                del positive, latent, sampled
                cleanup_segment_vram(enabled=clear_vram_between_chunks)
                log.info("H3 lip sync: chunk %d/%d video VAE decode starting", chunk.index + 1, len(chunks))
                decode_start = time.perf_counter()
                decoded = VAEDecode().decode(video_vae, {"samples": video})[0]
                log.info("H3 lip sync: chunk %d/%d video VAE decode finished in %.1fs",
                         chunk.index + 1, len(chunks), time.perf_counter() - decode_start)
                stop = chunk.context_frames + chunk.visible_frames
                if decoded.shape[0] < stop:
                    raise RuntimeError(f"Chunk {chunk.index + 1} decoded {decoded.shape[0]} frames; need {stop}.")
                visible = decoded[chunk.context_frames:stop].detach().cpu().float()
                log.info("H3 lip sync: chunk %d/%d appearance correction and stitching", chunk.index + 1, len(chunks))
                visible = appearance.process(visible)
                if output is None:
                    # One final allocation instead of retaining every chunk and
                    # torch.cat doubling peak RAM. IMAGE output still scales with duration.
                    output = torch.empty((total_frames, *visible.shape[1:]), dtype=torch.float32)
                output[chunk.start_frame:chunk.end_frame].copy_(visible)
                # Retain only the phase-aligned video tail for the next chunk.
                # Audio is always re-encoded from source, never from generated speech.
                from comfy.nested_tensor import NestedTensor
                from .h3_motion_context import steps_for_frames
                tail_steps = steps_for_frames(context_frames)
                if appearance.enabled and chunk.index + 1 < len(chunks):
                    # Feed corrected appearance back into the next generation.
                    # Prefix frames have already been removed; never filter or
                    # re-export the overlap twice. Re-encode only this short tail.
                    log.info("H3 lip sync: chunk %d/%d context-tail VAE encode (%d frames)",
                             chunk.index + 1, len(chunks), context_frames)
                    tail = video_vae.encode(visible[-context_frames:])
                    if tail.ndim != 5 or tail.shape[2] != tail_steps or tail.shape[3:] != video.shape[3:]:
                        raise RuntimeError("Corrected appearance tail does not match the H3 context grid.")
                    previous = {"samples": NestedTensor((tail.detach().cpu().clone(),))}
                    del tail
                else:
                    previous = {"samples": NestedTensor((video[:, :, -tail_steps:].detach().cpu().clone(),))}
                records.append({**asdict(chunk), "audio_start_frame": chunk.audio_start_frame,
                                "seed": chunk_seed, "prompt": chunk_prompt,
                                "first_frame_anchor_applied": bool(chunk.index and first_frame_anchor_strength)})
                progress.update_absolute(chunk.index + 1)
                del decoded, visible, video
                cleanup_segment_vram(enabled=clear_vram_between_chunks)
            from .lipsync_conditioning import build_refinement_conditioning
            mm.throw_exception_if_processing_interrupted()
            full_positive, full_negative = build_refinement_conditioning(
                frame_count=total_frames, video_vae=video_vae,
                clip=clip, prompt=prompt, width=width, height=height,
                generation_mode=generation_mode, first_frame=first_frame,
                ref_images=ref_images, ref_image_size=ref_image_size)
            progress.update_absolute(len(chunks) + 1)
            log.info("H3 lip sync: finished")
    finally:
        previous = None
        fixed_anchor = None
        cleanup_segment_vram(enabled=clear_vram_between_chunks)
    report = json.dumps({
        "mode": "long_audio_lipsync", "fps": FPS, "frame_count": total_frames,
        "audio_samples": waveform.shape[-1], "audio_sample_rate": rate,
        "audio_seconds": waveform.shape[-1] / rate, "video_seconds": total_frames / FPS,
        "audio_output": "original input, unchanged", "chunks": records,
        "generation_mode": generation_mode,
        "reference_image_count": len(ref_images) if ref_images else 1,
        "ref_image_size": ref_image_size if generation_mode == "ref2va" else None,
        "source_audio_locked": audio_denoise == 0 and audio_noise_mask is None,
        "reference_image_each_chunk": bool(reference_image_each_chunk or generation_mode == "ref2va"),
        "color_stabilization": float(color_stabilization),
        "detail_stabilization": float(detail_stabilization),
        "first_frame_anchor_strength": float(first_frame_anchor_strength),
        "first_frame_anchor_mode": "oldest hidden context keyframe; remaining motion context retained",
        "context_source": "corrected decoded tail" if appearance.enabled else "sampled latent tail",
        "refinement_frame_count": refinement_frame_count,
        "refinement_padding_frames": refinement_frame_count - total_frames,
        "export_conditioning": "global prompt and original image references; no chunk-local context",
        "export_chunk_prompt_suffixes": "not included; use global prompt for full-video refinement",
    }, indent=2, ensure_ascii=False)
    return output, audio, float(FPS), total_frames, report, full_positive, full_negative
