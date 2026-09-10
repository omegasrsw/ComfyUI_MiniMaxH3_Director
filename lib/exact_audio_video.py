"""H.264 video + unmodified floating-point PCM audio in Matroska."""

import math
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np
import torch

from .video_export import _ffmpeg_bin


def write_float_wav(path, audio):
    """Serialize AUDIO samples without integer quantization or resampling."""
    from ..director.lipsync import validate_audio

    waveform, rate = validate_audio(audio)
    bits = 64 if waveform.dtype == torch.float64 else 32
    channels, samples = waveform.shape[1:]
    block_align = channels * (bits // 8)
    size = samples * block_align
    if size + 36 > 0xffffffff:
        raise ValueError("Exact-audio WAV staging exceeds 4 GiB; split this recording before exporting.")
    dtype = torch.float64 if bits == 64 else torch.float32
    with Path(path).open("wb") as stream:
        stream.write(b"RIFF" + struct.pack("<I", size + 36) + b"WAVEfmt ")
        stream.write(struct.pack("<IHHIIHH", 16, 3, channels, rate, rate * block_align, block_align, bits))
        stream.write(b"data" + struct.pack("<I", size))
        for start in range(0, samples, 65536):
            block = waveform[0, :, start:start + 65536].detach().to(device="cpu", dtype=dtype).transpose(0, 1)
            stream.write(block.contiguous().numpy().astype(f"<f{bits // 8}", copy=False).tobytes())
    return bits


def write_exact_audio_video(path, images, audio, fps=24.0, check_interrupt=None):
    """Stream RGB frames to FFmpeg; copy the float PCM stream without encoding."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite.")
    if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[0] == 0
            or images.shape[-1] < 3 or min(images.shape[1:3]) < 2
            or images.shape[1] % 2 or images.shape[2] % 2):
        raise ValueError("Exact-audio video export needs nonempty IMAGE frames with even width/height.")
    ffmpeg = _ffmpeg_bin()
    if not ffmpeg:
        raise RuntimeError("Exact-audio video export requires FFmpeg (PATH or imageio-ffmpeg).")
    dest = Path(path)
    if dest.suffix.lower() != ".mkv":
        raise ValueError("Exact PCM audio export uses an .mkv container.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mmx_exact_audio_", dir=dest.parent) as temp:
        temp = Path(temp)
        bits = write_float_wav(temp / "audio.wav", audio)
        partial = temp / "video.mkv"
        with (temp / "ffmpeg.log").open("w+b") as errors:
            proc = subprocess.Popen([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{images.shape[2]}x{images.shape[1]}",
                "-r", str(float(fps)), "-i", "pipe:0", "-i", str(temp / "audio.wav"),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-crf", "18", "-c:a", "copy", str(partial),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
            try:
                for start in range(0, len(images), 8):
                    if check_interrupt:
                        check_interrupt()
                    block = images[start:start + 8, ..., :3].detach().cpu().float()
                    rgb = (block.clamp(0, 1) * 255).round().to(torch.uint8).numpy()
                    proc.stdin.write(np.ascontiguousarray(rgb).tobytes())
                proc.stdin.close()
                while proc.poll() is None:
                    if check_interrupt:
                        check_interrupt()
                    try:
                        proc.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        pass
            except BrokenPipeError as exc:
                proc.kill()
                proc.wait()
                errors.seek(0)
                raise RuntimeError(f"Exact-audio export failed: {errors.read().decode(errors='replace')}") from exc
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                if not proc.stdin.closed:
                    proc.stdin.close()
            if proc.returncode != 0:
                errors.seek(0)
                raise RuntimeError(f"Exact-audio export failed: {errors.read().decode(errors='replace')}")
        partial.replace(dest)
    return str(dest), bits
