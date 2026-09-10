# Ref2VA lip sync with exact source audio

Use **MiniMax H3 Director Ref2VA Audio Lip Sync** for a character, background,
clothing, props, or other image references, driven by a full speech recording.
The node generates overlapping chunks for the entire audio, keeping all image
references active alongside the previous generated video tail.

For long runs affected by increasing sharpness or color drift, load the
[stabilized example workflow](../example_workflows/minimax_h3_director_long_audio_lipsync_ref2va_stabilized.json).
It includes a connected appearance-image loader and enables both corrections
at `0.35`, using the same correction implementation as FL2VA.

[Load the example workflow](../example_workflows/minimax_h3_director_long_audio_lipsync_ref2va.json).
It uses ComfyUI core, this Director plugin, and FFmpeg for exact-audio export.
The existing FL2VA lip-sync node and workflows remain available.

## Setup

1. Load the updated plugin under `ComfyUI/custom_nodes/` and restart ComfyUI.
2. Load the example JSON. Select your H3 **ref2va** diffusion model, H3 video
   VAE, H3 audio VAE, and Qwen encoder with CLIP loader type **minimax**.
   The example uses the `h3/` model subfolders; adjust these paths if needed.
3. Upload the character to `ref_image_1` and the background to `ref_image_2`.
   Select the complete source recording in **Load Audio**. The example's
   `character.png`, `background.png`, and `speech.wav` are placeholders.
4. Describe each reference's role using `<Picture N>` tags, then queue once.
   No first-frame image or external loop is required.

For example:

```text
The character from <Picture 1> is in the setting shown in <Picture 2>,
speaking to camera with accurate lip sync to the supplied speech.
Preserve the character identity, clothing, background and lighting throughout.
Natural expressions and subtle gestures, static camera, no cuts.
```

The example uses 25 steps, `res_multistep`, `beta`, sigma shifts 12/3, a
10-second sampling cap, and 22 context frames. Use a LoRA only if it supports
the ref2va model; the FL2VA Turbo LoRA from the earlier workflow is not
automatically added here.

## References and appearance

- Up to **9 separate images**, connected consecutively as `ref_image_1` through
  `ref_image_9`, map directly to `<Picture 1>` through `<Picture 9>`. Each input
  must contain one image. Missing intermediate slots produce an error so
  reference numbering cannot silently change.
- Every chunk uses the official `MiniMaxH3ReferenceToVideo` node. The images
  enter both Qwen visual conditioning and the diffusion model's reference
  latents. The previous chunk's video context is added while retaining these
  references. Images are references rather than fixed opening/closing frames.
- `ref_image_size=match` scales references down toward the output pixel area;
  `max` can retain more detail and costs more memory/time. Native aspect ratios
  are retained. See the [official node implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_minimax_h3.py).
- This lip-sync node exposes image references. It does not accept voice/audio
  references that could compete with the supplied speech recording.
- Color/detail stabilization is optional and off by default. To enable it,
  connect `appearance_reference` containing the intended **whole scene**.
  That image supplies filter statistics only; it is not another `<Picture N>`
  reference. A face crop should not dictate the color of a different background.

The audio-driven frame planner, temporal overlap, final-frame crop, seed
progression and optional per-chunk prompt suffixes work as described in the
[long-audio guide](long-audio-lipsync.md). Avoid putting the full long transcript
in the global prompt: it is reused for every chunk. No ASR is run automatically.

## Countering color and sharpness drift

In the stabilized example, load a clean image of the intended **complete scene**
into **Appearance target: intended FULL scene**. Match the intended framing,
lighting and overall detail level. This is separate from character/background
references: it supplies fixed appearance statistics, not a forced first frame.
If one of your existing references already shows that full scene, connect it
to `appearance_reference` as well.

The same process runs in FL2VA and ref2va:

1. Keep the original image references active for every generated chunk.
2. Remove the generated overlap, then adjust the visible frames' color and
   contrast toward the fixed appearance reference using bounded RGB corrections.
3. Attenuate excess fine texture relative to that reference to counter growing
   oversharpening. Correction parameters are smoothed over time, retaining their
   history across chunk boundaries to limit abrupt changes.
4. Export those corrected frames and re-encode their last `context_frames`
   frames as the next chunk's motion context. This also applies the correction
   inside the generation loop, rather than only to the final exported video.

Start with `color_stabilization=0.35` and `detail_stabilization=0.35`, as wired
in the preset. Lower color strength if the grade changes too much; lower detail
strength if the image becomes too soft. Set either to `0` to disable it. These
are adjustable starting values, not settings validated for every scene.
The plain ref2va example keeps both at `0` for comparison.

The report shows `context_source: "corrected decoded tail"` when either filter
is active. The full source audio and its timing remain unchanged by this
process. Filters do not blend reference pixels into the mouth, reconstruct lost
detail, or repair identity/pose drift; stronger correction is not always better.

## What “exact audio” means

The source waveform is encoded into each chunk's audio latent with a **zero
denoise mask**. This ref2va node has no audio-denoise or audio-mask controls;
the model's generated audio is never decoded for output. Only video is decoded.
The node returns the original `AUDIO` object, with identical waveform sample
values, sample count, channels and sample rate. Resampling and mono-to-stereo
conversion occur only on private conditioning copies used by the H3 audio VAE.

The example connects `images`, `audio`, and `fps` to **MiniMax H3 Save Video
(Exact Audio)**. This writes:

- **MKV container**, with H.264 video;
- **floating-point PCM audio**, using the original sample rate and channels;
- float32 sample storage, or float64 for a float64 input; no integer
  quantization, normalization, mixing, audio resampling, or lossy audio codec.

The saver returns the file path under ComfyUI's output folder. Use a player
that supports MKV/PCM; browser preview support varies. The saver streams video
frames to FFmpeg and does not build another complete video buffer. PCM audio
uses more space than AAC. Audio staging currently supports recordings whose
uncompressed waveform fits within a standard 4-GiB WAV file.

“Exact” refers to the **decoded samples delivered by your audio loader**. An
MP3/AAC input has already been decoded before reaching the node. This saver
preserves those samples, not the original compressed file bytes or metadata.
The video has `ceil(audio_seconds * 24)` frames; the final frame can extend by
less than 1/24 second, while the entire audio remains unchanged.

You can instead use `CreateVideo` → `SaveVideo` for MP4. The source speech
content still comes from the input, but an AAC saver re-encodes it and cannot
promise sample-exact output. The Exact Audio saver also accepts outputs from
the existing FL2VA lip-sync node.

## Verification and limits

Tests check reference order, missing slots, references on every chunk, forced
audio locking, source-waveform equality, unchanged frame coverage, and MKV
round trips whose decoded PCM bytes exactly equal the input samples (float32
stereo and float64 mono, including non-integer durations).
The stabilized ref2va regression also checks that the exact corrected frames
exported at each boundary are re-encoded and passed into the following chunk,
while image references remain active and source audio stays bitwise unchanged.

`tests/smoke_lipsync_comfy.py` exercises both FL2VA and ref2va paths with real
ComfyUI conditioning, audio VAE weights, reference-plus-context PackedLayout,
and the H3 conditioning payload. Diffusion sampling, CLIP weights and the video
VAE are test doubles. A full ref2va generation is still needed to evaluate
visual reference adherence, seam quality and mouth synchronization for your
particular models, images and speech.

The complete IMAGE output remains in CPU RAM, proportional to recording
length. Reference tokens also increase sampling memory and time.
