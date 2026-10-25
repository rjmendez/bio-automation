"""BioCLIP 2 image embedder for taxon suggestions. GPU only: no CUDA is a RuntimeError, never a CPU fallback.

Why it exists: measured by embed_loo.py on the owner's 3,631 labelled iNat photos (same observation / same day excluded), a nearest-neighbour
vote over BioCLIP 2 vectors beats the SigLIP index at every taxonomic level (kingdom 93.5% vs 88.8%, genus 51.4% vs 30.9%, species 47.4% vs
29.8%). The SigLIP index (embed_index.Embedder, collection bio_photos) stays for bio_similar / triage_vision / lc_queue_from_raws.

Config is the ``bioclip`` object of embed_config.json: ``model`` (an open_clip model id, e.g. ``hf-hub:imageomics/bioclip-2``; never a literal here),
``collection`` (Qdrant collection of its vectors), ``cuda_device`` (int, or ``"auto"`` = the CUDA card with the most free VRAM, the default) and
optional ``sources`` ({source: folder} overriding embed_config.json's, e.g. when the iNat photo cache lives elsewhere).
``BIO_BIOCLIP_CUDA_DEVICE`` overrides ``cuda_device``. CUDA device order is not nvidia-smi order on this workstation: use the auto default or
check ``torch.cuda.get_device_name``.
"""
import os

import numpy as np

DEVICE_ENV = 'BIO_BIOCLIP_CUDA_DEVICE'


def pick_device(torch_mod, configured=None, environ=None):
    """'cuda:N'. The env var wins, then *configured*, then 'auto' (most free VRAM). RuntimeError when CUDA is not available or N does not exist."""
    environ = os.environ if environ is None else environ
    if not torch_mod.cuda.is_available():
        raise RuntimeError('bioclip_embed: CUDA is not available to torch; this stage runs on the GPU only and does not fall back to CPU')
    n = torch_mod.cuda.device_count()
    raw = environ.get(DEVICE_ENV)
    if raw in (None, ''):
        raw = configured
    if raw in (None, '', 'auto'):
        free = [torch_mod.cuda.mem_get_info(i)[0] for i in range(n)]
        return f'cuda:{max(range(n), key=lambda i: free[i])}'
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        raise RuntimeError(f'bioclip_embed: cuda device {raw!r} is not an integer or "auto"') from None
    if not 0 <= idx < n:
        raise RuntimeError(f'bioclip_embed: cuda device {idx} requested but torch sees {n} CUDA device(s)')
    return f'cuda:{idx}'


class BioClipEmbedder:
    """embed(paths) -> float32 (N, dim) array of L2-normalised BioCLIP 2 image vectors, same contract as embed_index.Embedder."""

    def __init__(self, cfg, torch_mod=None, open_clip_mod=None, load_image=None, environ=None):
        if torch_mod is None:
            import torch as torch_mod
        if open_clip_mod is None:
            import open_clip as open_clip_mod
        if load_image is None:
            from PIL import Image
            load_image = lambda p: Image.open(p).convert('RGB')  # noqa: E731
        self.torch = torch_mod
        self.load_image = load_image
        self.model_id = cfg['model']
        self.device = pick_device(torch_mod, cfg.get('cuda_device'), environ)
        model, _, self.preprocess = open_clip_mod.create_model_and_transforms(self.model_id)
        self.model = model.to(self.device).half().eval()
        self.open_clip = open_clip_mod
        self._tokenizer = None

    def encode_text(self, strings, batch=1024):
        """float16 (N, dim) array of L2-normalised text vectors for *strings*, in order (the tokenizer is loaded on first use)."""
        t = self.torch
        if self._tokenizer is None:
            self._tokenizer = self.open_clip.get_tokenizer(self.model_id)
        out = []
        with t.no_grad():
            for s in range(0, len(strings), batch):
                tok = self._tokenizer(list(strings[s:s + batch])).to(self.device)
                v = self.model.encode_text(tok).float().cpu().numpy()
                out.append((v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)).astype(np.float16))
        return np.concatenate(out)

    def logit_scale(self):
        """The model's learned temperature (CLIP's logit_scale, exponentiated): softmax(logit_scale * cosine) is the zero-shot class probability."""
        return float(self.model.logit_scale.detach().exp())

    def embed(self, paths):
        t = self.torch
        x = t.stack([self.preprocess(self.load_image(p)) for p in paths]).to(self.device).half()
        with t.no_grad():
            v = self.model.encode_image(x).float().cpu().numpy()
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
