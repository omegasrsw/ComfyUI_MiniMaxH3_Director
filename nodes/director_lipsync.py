"""Audio-driven alternative to the Director's manually authored timeline."""

from ..director.lipsync import generate_lipsync


class MiniMaxH3DirectorLongAudioLipSync:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers

        return {
            "required": {
                "model": ("MODEL",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "clip": ("CLIP",),
                "audio": ("AUDIO",),
                "first_frame": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default":
                    "A continuous shot of the person speaking to camera, with accurate lip sync "
                    "to the supplied speech, natural facial expressions and subtle gestures. "
                    "Preserve appearance, lighting and framing. Static camera, no cuts."}),
                "width": ("INT", {"default": 864, "min": 32, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 480, "min": 32, "max": 4096, "step": 32}),
                "chunk_seconds": ("FLOAT", {"default": 10.0, "min": 5.17, "max": 15.0, "step": 0.01,
                    "tooltip": "Maximum sampled duration INCLUDING context; rounded down to H3's frame grid."}),
                "context_frames": ([5, 22, 39, 56], {"default": 22}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 1000}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 100.0}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0}),
                "audio_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "0 locks source audio while sampling video. Higher values allow audio to change internally."}),
                "clear_vram_between_chunks": ("BOOLEAN", {"default": True,
                    "tooltip": "Unload models before each video VAE decode and between chunks. Recommended for limited VRAM; disabling may reduce reload overhead when everything fits."}),
            },
            "optional": {
                "sigmas": ("SIGMAS", {"tooltip": "Overrides steps/scheduler. Use the same H3 shifts on the scheduler's model."}),
                "audio_noise_mask": ("MASK", {"tooltip": "Overrides audio_denoise; 1 frame or ceil(audio seconds * 24) frames. Spatial maximum becomes audio-time strength."}),
                "chunk_prompts": ("STRING", {"multiline": True, "default": "",
                    "tooltip": "Optional JSON array, one prompt suffix per chunk. Include overlap speech in each suffix; see docs/long-audio-lipsync.md."}),
                "reference_image_each_chunk": ("BOOLEAN", {"default": True,
                    "tooltip": "Keep the original portrait in the text/vision conditioning of every chunk. Motion still comes from the previous tail."}),
                "color_stabilization": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Optional bounded color/contrast matching to the input image. Try 0.35 for a static talking shot. Corrected tail is re-encoded as context."}),
                "detail_stabilization": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Optional attenuation of excess fine texture relative to the input. Try 0.35. Can soften detail; does not restore identity."}),
                "first_frame_anchor_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "EXPERIMENTAL: original image latent in the first HIDDEN context keyframe of every continuation chunk. 0=old behavior, 1=full original anchor; intermediate values blend latents. Remaining context keeps recent motion. Can pull pose/framing toward the original. Compare 1 versus 0 with a fixed seed."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "report", "positive", "negative")
    OUTPUT_TOOLTIPS = (
        "Complete stitched video, after optional appearance correction.",
        "Original input audio, unchanged.", "24 fps.", "Exact visible frame count; trim decoded upscale output to this length.",
        "Chunk timing and required frame-grid padding for optional external VAE encoding.",
        "Full-video global prompt and image guidance. Ref2VA keeps reference blocks; FL2VA keeps vision embeddings without its fixed-resolution first-frame anchor. No chunk-local context/suffixes.",
        "Empty conditioning: the generator uses CFG 1 with no negative prompt. Use BasicGuider or CFG 1 downstream.",
    )
    FUNCTION = "execute"
    CATEGORY = "MiniMax H3/Director"
    DESCRIPTION = (
        "Generate a continuous lip-sync video for the full input audio. Uses H3 fl2va, "
        "source audio latents and previous video-tail context at 24 fps. Outputs the "
        "original soundtrack and an exact timeline report. Final images use CPU RAM "
        "proportional to duration. Outputs global positive/empty negative conditioning. "
        "Encode corrected images externally with the H3 video VAE when refinement is needed."
    )

    def execute(self, **kwargs):
        return generate_lipsync(**kwargs)
