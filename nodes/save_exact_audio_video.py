"""Lossless audio export companion for both lip-sync generators."""

from ..lib.exact_audio_video import write_exact_audio_video


class MiniMaxH3SaveVideoExactAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",), "audio": ("AUDIO",),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            "filename_prefix": ("STRING", {"default": "video/MiniMaxH3_ExactAudio"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/Director"
    DESCRIPTION = (
        "Save MKV with H.264 video and lossless float PCM audio. Preserves the supplied AUDIO "
        "sample values, channels and rate; no normalization, resampling or AAC encoding. "
        "The audio loader has already decoded compressed input files. File path is returned; "
        "use a player supporting MKV/PCM (browser playback may be unavailable)."
    )

    def save(self, images, audio, fps, filename_prefix):
        from pathlib import Path
        import folder_paths
        import comfy.model_management as mm

        folder, name, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory(), images.shape[2], images.shape[1])
        path = Path(folder) / f"{name}_{counter:05}_.mkv"
        saved, _ = write_exact_audio_video(path, images, audio, fps,
                                          check_interrupt=mm.throw_exception_if_processing_interrupted)
        return (saved,)
