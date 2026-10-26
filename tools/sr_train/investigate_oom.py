"""Local, free (CPU, no CUDA OOM risk) investigation of the AWS SAM3 training OOM.

Question: is the crash a genuinely huge-but-fixed per-forward-pass memory need
(architecture is just big at 1008px), or an accumulation/leak (memory keeps
growing across encoder layers/steps without being freed, so "bigger GPU" would
only delay, not fix, the crash)?

Builds the REAL model from the REAL local pilot config and pulls ONE REAL batch
from the REAL training dataloader (same objects trainer.py uses) -- no
synthetic stand-ins -- then runs a single forward pass on CPU with:
  1. A patch on torch.nn.functional._in_projection_packed (the exact crash line
     in every AWS OOM) that prints query/key/value shapes and element counts.
  2. Forward hooks on backbone, geometry_encoder, transformer.encoder, and each
     individual encoder layer, printing process RSS (psutil) before/after each
     -- growth that never comes back down between layers would mean a leak;
     growth that's consistently large per layer (architecture-sized) would not.

Run with F:\\venvs\\sam3\\Scripts\\python.exe (has sam3 + its training deps installed).
"""
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent / "sam3_vendor")  # configs are package-relative
sys.path.insert(0, str(Path(__file__).resolve().parent / "sam3_vendor"))

import psutil  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from hydra import compose, initialize_config_module  # noqa: E402
from hydra.utils import instantiate  # noqa: E402

from sam3.train.train import register_omegaconf_resolvers  # noqa: E402

_PROC = psutil.Process()


def _rss_mb() -> float:
    return _PROC.memory_info().rss / (1024 * 1024)


_call_count = [0]
_orig_in_proj = F._in_projection_packed


def _patched_in_projection_packed(query, key, value, w, b):
    _call_count[0] += 1
    n = _call_count[0]
    q_elems = query.numel()
    print(
        f"[_in_projection_packed call #{n}] query.shape={tuple(query.shape)} "
        f"({q_elems:,} elems, {q_elems * 4 / 1e6:.1f} MB as fp32) "
        f"key.shape={tuple(key.shape)} value.shape={tuple(value.shape)} "
        f"w.shape={tuple(w.shape)} | RSS={_rss_mb():.0f} MB",
        flush=True,
    )
    return _orig_in_proj(query, key, value, w, b)


F._in_projection_packed = _patched_in_projection_packed

# Debug-only: position_encoding.py's precompute cache hardcodes device="cuda"
# (torch.zeros((1, 1) + size, device="cuda")) regardless of where the rest of the
# model lives -- fine on the real AWS run (everything really is on cuda there), but
# this script deliberately forces CPU, so redirect it instead of patching vendor code.
_orig_zeros = torch.zeros


def _cpu_forced_zeros(*args, **kwargs):
    if kwargs.get("device") == "cuda":
        kwargs["device"] = "cpu"
    return _orig_zeros(*args, **kwargs)


torch.zeros = _cpu_forced_zeros

# Print the cross-attention "memory" (image feature) sequence length right before
# TransformerEncoderLayer.forward_pre uses it -- this is the actual question: how
# long is the key sequence a prompt's small query cross-attends against?
from sam3.model.encoder import TransformerEncoderLayer  # noqa: E402

_orig_forward_pre = TransformerEncoderLayer.forward_pre


def _patched_forward_pre(self, tgt, memory, *args, **kwargs):
    print(
        f"  [forward_pre] tgt.shape={tuple(tgt.shape)} memory.shape={tuple(memory.shape)} "
        f"({memory.numel():,} elems, {memory.numel() * 4 / 1e6:.1f} MB as fp32) | RSS={_rss_mb():.0f} MB",
        flush=True,
    )
    return _orig_forward_pre(self, tgt, memory, *args, **kwargs)


TransformerEncoderLayer.forward_pre = _patched_forward_pre


def _hook(label):
    def pre(module, input):
        print(f"  -> entering {label} | RSS={_rss_mb():.0f} MB", flush=True)

    def post(module, input, output):
        print(f"  <- leaving  {label} | RSS={_rss_mb():.0f} MB", flush=True)

    return pre, post


def main():
    register_omegaconf_resolvers()
    initialize_config_module("sam3.train", version_base="1.2")
    cfg = compose(config_name="configs/custom/weak_categories_pilot.yaml")

    print(f"RSS before model build: {_rss_mb():.0f} MB")
    model = instantiate(cfg.trainer.model, _convert_="all")
    model.train()  # matches the real training call path (activation checkpointing is self.training-gated)
    print(f"RSS after model build:  {_rss_mb():.0f} MB")

    pre, post = _hook("backbone")
    model.backbone.register_forward_pre_hook(lambda m, i, pre=pre: pre(m, i))
    model.backbone.register_forward_hook(lambda m, i, o, post=post: post(m, i, o))
    pre, post = _hook("geometry_encoder")
    model.geometry_encoder.register_forward_pre_hook(lambda m, i, pre=pre: pre(m, i))
    model.geometry_encoder.register_forward_hook(lambda m, i, o, post=post: post(m, i, o))
    pre, post = _hook("transformer.encoder (whole)")
    model.transformer.encoder.register_forward_pre_hook(lambda m, i, pre=pre: pre(m, i))
    model.transformer.encoder.register_forward_hook(lambda m, i, o, post=post: post(m, i, o))
    for idx, layer in enumerate(model.transformer.encoder.layers):
        pre, post = _hook(f"transformer.encoder.layers[{idx}]")
        layer.register_forward_pre_hook(lambda m, i, pre=pre: pre(m, i))
        layer.register_forward_hook(lambda m, i, o, post=post: post(m, i, o))

    # DistributedSampler (used even for a single process, matching single_proc_run's
    # pattern) needs a process group -- gloo loopback, world_size=1.
    import tempfile
    import torch.distributed as dist
    # TCPStore (even with an explicit 127.0.0.1 init_method) still tries to bind by
    # hostname internally and Windows refuses it (system error 10013) -- same class of
    # Windows-socket gotcha as the gloo hostname bug train.py already works around.
    # FileStore sidesteps sockets entirely.
    store_file = Path(tempfile.gettempdir()) / "sam3_investigate_oom.store"
    store_file.unlink(missing_ok=True)
    dist.init_process_group(
        backend="gloo", rank=0, world_size=1,
        init_method=f"file:///{store_file.as_posix()}",
    )

    print("\nBuilding the real training dataset + dataloader...")
    train_dataset = instantiate(cfg.trainer.data.train)
    dataloader = train_dataset.get_loader(epoch=0)
    print(f"RSS after dataloader build: {_rss_mb():.0f} MB")

    print("\nPulling one real batch...")
    raw_batch = next(iter(dataloader))
    print(f"RSS after one batch pulled: {_rss_mb():.0f} MB")

    # Exactly trainer.py's _step(): the dataloader yields {key: BatchedDatapoint},
    # popitem() the actual object, then move it to the compute device (CPU here).
    from sam3.model.utils.misc import copy_data_to_device
    _, batch = raw_batch.popitem()
    batch = copy_data_to_device(batch, torch.device("cpu"), non_blocking=False)

    print("\n=== Running ONE real forward pass on CPU ===\n")
    with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        # matches cfg.trainer.optim.amp (enabled=True, amp_dtype=bfloat16) -- AMP changes
        # which ops run in bf16 vs fp32, so it matters for realistic memory figures too.
        model(batch)
    print(f"\nRSS after forward pass: {_rss_mb():.0f} MB")
    print(f"Total _in_projection_packed calls: {_call_count[0]}")


if __name__ == "__main__":
    main()
