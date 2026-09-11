# Long audio lip sync

For the **ref2va** model with multiple character/background images and exact
PCM audio export, see [Ref2VA lip sync](ref2va-lipsync.md).

`MiniMax H3 Director Long Audio Lip Sync` generates video for the entire input
audio automatically, using the fl2va model and the previous generated video
tail as temporal context. It is a separate node in this Director package;
the existing timeline Director retains its existing behavior.

Both lip-sync nodes also output one **whole-video** `latent`, `positive`, and
`negative` for downstream upscaling/refinement. The video latent is re-encoded
from the complete stitched, corrected frames. It is not the final generation
chunk. See [whole-video export and wiring](../README.md#whole-video-latent-outputs-for-upscaling),
including terminal-padding removal, global conditioning semantics and memory cost.

## Load a workflow

- [Base model workflow](../example_workflows/minimax_h3_director_long_audio_lipsync.json):
  only ComfyUI core and this plugin; 25 steps, `res_multistep`, `simple`.
- [Adapted Turbo workflow](../example_workflows/minimax_h3_director_long_audio_lipsync_turbo.json):
  model paths, Turbo LoRA, attention backend, feed-forward chunking, 8 steps,
  `er_sde`, `beta`, and 12/3 sigma shifts from the supplied
  `minimax h3 i2v custom audio_Dev.json`. Requires the same KJNodes and Turbo LoRA.

1. Make this updated repository available under `ComfyUI/custom_nodes/`, then
   restart ComfyUI and drag in one of the JSON files. If this is a development
   checkout elsewhere, ComfyUI must load this checkout or an updated copy.
2. Select your model files (the examples use the supplied workflow's `h3/`
   subfolders), portrait in **Load Image**, and full recording in **Load Audio**.
   The example image/audio filenames are placeholders from the supplied workflow.
3. Use a prompt describing the person speaking, expressions, gestures, camera,
   and appearance. Set width/height in multiples of 32. `clip` must use the
   `minimax` loader type, with the H3 video and audio VAEs on their respective ports.
4. Start with `chunk_seconds=10`, `context_frames=22`, `audio_denoise=0`.
   Queue once. `CreateVideo` → `SaveVideo` saves the combined video with source audio.

An existing workflow can feed its patched `MODEL`, `CLIP`, two `VAE`s, resized
portrait `IMAGE`, full `AUDIO`, and visual prompt into this node. It replaces
the single-clip conditioning, separate/concat AV latent, sampling, and decode
branch. The node applies the H3 sigma shifts itself. An optional external
`sigmas` input overrides steps/scheduler; calculate it with a model using the
same shift settings. No external loop nodes are needed.

## Timing and continuity

The audio is the master clock at a fixed **24 fps**. The node exports exactly
`ceil(audio_samples * 24 / sample_rate)` frames. The last video frame may
extend beyond the audio by less than 1/24 second; audio samples are unchanged.

`chunk_seconds` caps the entire sampled window, **including its context**.
It rounds down to H3's `17k+5` frame grid, with a maximum of 15 seconds.
For 10 seconds / 22 context frames:

| Chunk | Sampled source interval (frames) | Output interval (frames) | Sample length |
|---|---|---|---|
| 1 | 0–226 | 0–226 | 226 |
| 2 | 204–430 | 226–430 | 226 |
| 3 | 408–634 | 430–634 | 226 |

Intervals are half-open. The second chunk sees the previous 22 video frames
and source audio starting at frame 204. Its first 22 decoded frames are
discarded; output begins at frame 226. There are no duplicated or dropped
output frames at the join. These regular chunk sizes keep the reused video
latent tail aligned with the VAE's temporal phase.

The final sample rounds up to the H3 grid, with a minimum of 124 sampled
frames. Source audio past EOF is silence for conditioning; the final decoded
video is cropped to the remaining real duration. Even a very short input
therefore incurs a minimum H3 generation window.

Video context uses this repo's existing motion-context keyframe/layout
implementation. Only the compact previous video latent tail is retained for
the next generation. Audio for the **whole** sample window, including the
overlap, is encoded from the original waveform. Generated speech is never
substituted as the next chunk's source. Absolute sample offsets prevent
per-chunk audio rounding errors from accumulating.

## Audio and prompts

`audio_denoise=0` freezes the supplied audio latent while video is sampled.
Higher values let the model alter audio internally and can weaken adherence
to the original speech. The output `audio` is always the original input,
irrespective of this setting; the generated audio is not decoded or concatenated.
Mono is duplicated to stereo for H3 conditioning, but returned audio keeps
its original channels and sample rate. Only one mono/stereo recording is
accepted per node execution.

The supplied workflow's amplitude-circle mask is optional here. Connect a
`MASK` to `audio_noise_mask` to override `audio_denoise`. It must contain one
mask or exactly `ceil(audio_seconds * 24)` masks. Each mask's spatial maximum
becomes one audio-denoise strength, mapped by timestamp to H3's 40 Hz audio
grid. This is an explicit temporal interpretation; a circle's image coordinates
do not represent positions in audio. Leaving it disconnected locks all speech.

The global `prompt` is reused for every chunk. Avoid putting the entire long
transcript there, since that asks each chunk to speak the entire recording.
This node does **not** run ASR or the Qwen prompt enhancer automatically.
Your existing image-description prompt can feed `prompt`; timed dialogue can
be supplied through `chunk_prompts` as a JSON array of suffixes, one per
generated chunk. Each suffix should describe its whole sampled audio window,
including overlap dialogue; timestamps within a suffix should be relative to
that window. The global visual prompt is prepended to each suffix.

The number and windows can be calculated before generation from the source
audio length using the planner (run from the repository with Python + PyTorch):

```python
from director.lipsync import plan_audio_chunks

for c in plan_audio_chunks(sample_count=32000 * 60, sample_rate=32000,
                           chunk_seconds=10, context_frames=22):
    print(c.index + 1, c.audio_start_frame / 24,
          (c.audio_start_frame + c.sample_frames) / 24)
```

The `report` output includes every sampled/exported interval, per-chunk seed
(`seed + index`, wrapping at 64 bits), and prompt. Invalid prompt counts fail
before model inference. Transcript alignment remains the responsibility of
the upstream transcription/prompt workflow.

## Appearance drift over successive chunks

For a stronger original-image signal in continuation chunks, see the new
[experimental hidden first-frame anchor](../README.md#experimental-original-image-anchor-in-every-continuation).
`first_frame_anchor_strength=1` restores the fixed original latent at the oldest
hidden context keyframe; `0` keeps the old path. It retains the other motion
blocks, overlap trim and audio clock. Compare both with the same fixed seed;
the visual improvement is not established by the wiring tests.

`reference_image_each_chunk` now defaults to **true**. Previously only the
first chunk sent the portrait into the text/vision encoder; later chunks
inherited appearance solely through the generated video tail. Each new
generation could therefore reinforce color, contrast, texture, and identity
errors from its predecessor.

With this option on, the original portrait is supplied to the text/vision
encoder every time. Its first-frame latent anchor is replaced by the moving
context tail, while its visual embeddings remain in the conditioning. The
portrait guides appearance and the tail guides motion. This is a semantic
appearance reference, not a pixel-perfect identity lock. Turn the option off
and leave the two stabilization sliders at zero to reproduce the old path.

Two optional filters provide additional control for static talking shots:

| Setting | Initial trial value | Effect |
|---|---|---|
| `reference_image_each_chunk` | true | Original portrait in every chunk's vision conditioning |
| `color_stabilization` | 0.35 | Bounded RGB mean/contrast correction toward the input image |
| `detail_stabilization` | 0.35 | Attenuates excess fine texture relative to the input image |

Both sliders default to **0 (off)** so existing workflows do not silently
receive color grading or softening. Color correction limits per-channel gain
to 0.85–1.15 and offset to ±0.08 before applying the slider strength. Detail
correction attenuates a 3×3 high-pass band by at most 50% at full strength;
it never amplifies that band. This can reduce harsh skin/hair texture, but
does not reconstruct lost detail or restore facial geometry.

Correction parameters are smoothed over roughly half a second and carried
across chunk boundaries. Only visible frames are filtered. When a slider is
nonzero, the corrected output's final context frames are re-encoded for the
next chunk, so the model receives the corrected appearance too. This adds one
short video-VAE encode per handoff and uses the same context length and
audio clock. The original audio remains unchanged. With both sliders off,
the direct latent-tail path is retained.

These global appearance statistics suit stable lighting and composition.
They can suppress intentional lighting/color changes or soften legitimate
detail after camera motion; reduce or disable the sliders for such shots.
They also cannot undo a change in facial proportions or framing. Try the
portrait reference alone first, then add moderate correction if needed.

Check the portrait preprocessing as well. In the supplied 25-second example,
the workflow enlarged the portrait 4× with RTX VSR, then resized it with
`nearest-exact`, and stretched its 735×860 aspect ratio to a 1024×1024 square.
That pipeline can introduce hard edges and changes the portrait's geometry
before H3 sees it. For a useful comparison, bypass the 4× enhancement and use
a single Lanczos/bilinear resize; choose an output aspect close to the original
(for example 768×896), or crop deliberately rather than stretch.

If drift persists, compare the core-only base-model workflow with the Turbo
workflow on the same short excerpt. The example uses Turbo strength 1 with
8 steps; this does not establish whether Turbo causes the observed drift.
Do not assume that increasing steps on a distilled model will improve it.
Visual quality from the new conditioning still needs an actual multi-chunk
diffusion render; the automated smoke test checks wiring and timing only.

## Memory and limits

Sampling VRAM is bounded by one chunk plus its context. The final `IMAGE`
tensor resides in CPU RAM and grows with audio duration: approximately
6.7 GiB per minute at 864×480 in float32, before downstream video-encoding
overhead. The node preallocates a single final frame buffer to avoid a second
full-size concatenation allocation. This is not a streaming disk renderer.

Whole-video latent export adds a video-VAE encode after all frames are stitched.
It reserves terminal padding in the existing frame allocation and retains the
complete encoded latent too; export memory and work grow with the full duration.
`export_refinement=true` enables this in both modes. Set it to `false` for
video/audio-only output; disconnect refinement consumers because the latent
output becomes `None` and both conditioning outputs become empty lists.

Export passes at most 73 frames to each VAE call: 68 core frames plus five
lookahead frames. Starts remain on the global 17-frame clip grid; each interior
window keeps its first 20 latent tokens, and only the last window keeps the
terminal tokens. This preserves the stock H3 encoder's independent-clip layout
and single final three-token drop. It bounds full-image normalization/dtype
copies and temporary encoded tensors without changing the stitched timeline.
The full IMAGE buffer and final CPU latent still grow with duration. Terminal
messages identify each generation phase and every completed export window.

Before each video VAE decode, both lip-sync modes release the chunk's sampling
inputs, masks, audio latent and conditioning, and retain only its sampled video
latent on CPU. `clear_vram_between_chunks=true` also unloads models at this
handoff and between chunks using the existing Director cleanup helper. This
gives decoding more VRAM, but adds model reload overhead. Disable it if keeping
models resident fits your GPU and improves throughput. The whole-video latent
and conditioning are exported to CPU after all chunks finish; their final
encode is a separate phase, not an accumulating GPU cache during chunk decoding.
ComfyUI cancellation is checked at each chunk
and remains available through its sampler. This mode reruns all chunks after
an interrupted/changed queue; it does not use the timeline Director's disk
cache, run-selection, or Refine node.

Context gives the model recent appearance and motion information; it cannot
guarantee invisible seams, perfect lip sync, or identity preservation over
arbitrarily long recordings. Validate a multi-chunk excerpt with your portrait,
prompt, and model settings before a large render.

## Verification

`python -m unittest discover -s tests -v` checks audio/frame coverage, context
phase alignment, non-integer durations, EOF padding, mask timing, seed wrap,
cancellation, and multi-chunk output assembly using lightweight model doubles.

With an installed ComfyUI environment:

```text
python tests/smoke_lipsync_comfy.py /path/to/ComfyUI /path/to/minimax_h3_audio_vae_fp32.safetensors
```

This imports the registered plugin, uses the real official conditioning/audio
nodes, H3 audio VAE weights, resampling, NestedTensor, and motion-context
PackedLayout path across three chunks. CLIP, video VAE, and diffusion sampling
remain test doubles; this check does not measure generated visual quality.
