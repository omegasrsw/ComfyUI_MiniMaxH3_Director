"""Optional real H3 VAE + third-party upscaler smoke, with no diffusion.

Usage: python tests/smoke_lipsync_export.py COMFY_ROOT UPSCALER_PY VIDEO_VAE_WEIGHTS
Uses the installed minimax_h3_latent_upscaler_3d_bf16.safetensors on CUDA.
"""

import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types

root = Path(__file__).resolve().parents[1]
comfy_root, upscaler_path, vae_path = sys.argv[1:4]
sys.argv = sys.argv[:1]
sys.path.insert(0, comfy_root)
import torch
import comfy.sd
import comfy.utils
import comfy.model_management as mm
from nodes import VAEDecode

package = types.ModuleType('mmx_export_smoke')
package.__path__ = [str(root)]
sys.modules[package.__name__] = package
export = importlib.import_module('mmx_export_smoke.director.lipsync_export')
spec = importlib.util.spec_from_file_location('h3_upscale_smoke', upscaler_path)
upscaler = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = upscaler
spec.loader.exec_module(upscaler)


class Clip:
    def tokenize(self, prompt, images=None, minimax_ref_items=None):
        assert prompt == 'A person speaking.'
        assert len(images or minimax_ref_items) == (1 if images is not None else 2)
        return prompt

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.ones(1, 7, 8), {}]]


vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path, safe_load=True))
# More than 15 seconds, not aligned to the 17k+5 grid. Terminal repeats only.
count, padded = 421, 430
frames = torch.linspace(0.1, 0.8, count).view(-1, 1, 1, 1).expand(-1, 32, 32, 3).clone()
frames = torch.cat((frames, frames[-1:].expand(padded-count, -1, -1, -1)))
reference = frames[:1].clone()
with torch.inference_mode():
    latent, positive, negative = export.export_stitched_video(
        frames=frames, frame_count=count, video_vae=vae, clip=Clip(),
        prompt='A person speaking.', width=32, height=32, generation_mode='ref2va',
        ref_images={'ref_image_1': reference, 'ref_image_2': reference})
    assert tuple(latent['samples'].shape) == (1, 24, 127, 2, 2)
    assert torch.isfinite(latent['samples']).all() and negative == []
    assert len(positive[0][1]['minimax_refs']) == 2
    assert not positive[0][1].get('minimax_keyframes')
    mm.unload_all_models()
    result = upscaler.MinimaxH3LatentUpscaler3D.execute(
        latent=latent, model_name='minimax_h3_latent_upscaler_3d_bf16.safetensors',
        mode={'mode': upscaler.UpscaleMode.SCALE_BY, 'scale': 2.0}, align=32,
        enable_chunking=True, device='cuda', precision='bf16')
    enlarged = result.args[0]
    assert tuple(enlarged['samples'].shape) == (1, 24, 127, 4, 4)
    assert torch.isfinite(enlarged['samples']).all()
    decoded = VAEDecode().decode(vae, enlarged)[0]
    assert tuple(decoded.shape) == (padded, 64, 64, 3), tuple(decoded.shape)
    assert len(decoded[:count]) == count
print(json.dumps({'status': 'passed', 'real_video_vae': True, 'real_chunked_upscaler': True,
                  'source_frames': count, 'padded_frames': padded,
                  'input_shape': list(latent['samples'].shape),
                  'upscaled_shape': list(enlarged['samples'].shape),
                  'decoded_shape': list(decoded.shape)}))
