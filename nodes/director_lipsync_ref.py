"""Reference-to-video lip sync with mandatory source-audio preservation."""

from .director_lipsync import MiniMaxH3DirectorLongAudioLipSync
from ..director.lipsync import generate_lipsync


class MiniMaxH3DirectorRefAudioLipSync(MiniMaxH3DirectorLongAudioLipSync):
    @classmethod
    def INPUT_TYPES(cls):
        schema = super().INPUT_TYPES()
        schema["required"] = {
            ("ref_image_1" if key == "first_frame" else key): value
            for key, value in schema["required"].items() if key != "audio_denoise"
        }
        schema["required"]["ref_image_1"] = ("IMAGE", {"tooltip": "Character or main reference → <Picture 1>. References guide content, not a fixed opening frame."})
        schema["required"]["prompt"] = ("STRING", {"multiline": True, "default":
            "A continuous shot of the character in <Picture 1>, speaking to camera with accurate "
            "lip sync to the supplied speech. Preserve the character's appearance. Natural expressions "
            "and subtle gestures, static camera, no cuts."})
        schema["required"]["scheduler"] = (schema["required"]["scheduler"][0], {"default": "beta"})
        for key in ("audio_noise_mask", "reference_image_each_chunk"):
            schema["optional"].pop(key)
        schema["optional"].update({
            f"ref_image_{i}": ("IMAGE", {"tooltip": f"Reference for <Picture {i}>: background, clothing, prop, or additional character view. Fill slots consecutively."})
            for i in range(2, 10)
        })
        schema["optional"]["ref_image_size"] = (["match", "max"], {"default": "match",
            "tooltip": "match: scale reference area down to output area. max: retain more reference detail, using more VRAM and time."})
        schema["optional"]["appearance_reference"] = ("IMAGE", {"tooltip":
            "Required only when color/detail stabilization is enabled. Use an image of the intended full scene; a character crop can bias the background's colors."})
        schema["optional"]["color_stabilization"][1]["tooltip"] = (
            "Match color/contrast toward appearance_reference with bounded, temporally smoothed correction. "
            "Try 0.35. The corrected video tail feeds the next chunk; source audio stays unchanged.")
        schema["optional"]["detail_stabilization"][1]["tooltip"] = (
            "Reduce excess fine texture relative to appearance_reference. Try 0.35; lower if too soft. "
            "The corrected video tail feeds the next chunk. Does not restore lost detail or identity.")
        # Append the new widget after ref_image_size to preserve old widget order.
        anchor = schema["optional"].pop("first_frame_anchor_strength")
        schema["optional"]["first_frame_anchor_strength"] = (anchor[0], {**anchor[1], "tooltip":
            "EXPERIMENTAL: use appearance_reference (full scene required) as the oldest hidden context keyframe in each continuation. 0=off, 1=full anchor; values between blend latents. Remaining motion context and image references stay active. May pull pose/framing toward the scene reference."})
        schema["optional"]["export_refinement"] = schema["optional"].pop("export_refinement")
        return schema

    DESCRIPTION = (
        "Long-audio lip sync using the H3 ref2va model. Up to nine image references remain active "
        "on every chunk together with the previous video tail. Source audio is locked during "
        "sampling and returned unchanged, with its original samples, channels and sample rate. "
        "Optional color/detail stabilization uses a full-scene appearance reference and feeds the "
        "corrected tail into subsequent chunks, as in the FL2VA lip-sync node. "
        "Use <Picture N> tags to assign reference roles. No generated soundtrack is exported."
        " Outputs one whole-video latent after stitching/correction, global reference conditioning, "
        "and empty negative conditioning for downstream upscale/refinement."
    )

    def execute(self, ref_image_1, **kwargs):
        refs = {"ref_image_1": ref_image_1}
        for i in range(2, 10):
            refs[f"ref_image_{i}"] = kwargs.pop(f"ref_image_{i}", None)
        return generate_lipsync(generation_mode="ref2va", ref_images=refs,
                                audio_denoise=0.0, audio_noise_mask=None, **kwargs)
