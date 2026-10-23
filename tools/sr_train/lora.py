"""Minimal from-scratch LoRA linear wrapper for fine-tuning the vendored HAT-S
backbone (hat_vendor/hat_arch.py) without an external peft dependency -- HAT's
module layout (HAB/OCAB inside RHAG/AttenBlocks) is custom enough that peft's
auto-detection would not target the right layers anyway.

LoRA adds a low-rank update to a frozen nn.Linear: out = base(x) + scaling *
dropout(x) @ A^T @ B^T, where A is [r, in_features] (Kaiming-initialized) and B
is [out_features, r] (zero-initialized). B starting at zero means the wrapped
layer is numerically IDENTICAL to the frozen base layer the moment it is
inserted -- the already-measured HAT-S zero-shot numbers are not perturbed
before a single training step runs.

Insertion points (see train_lora_hat.py for why): the MLP linears inside every
HAB block, and the OCAB module's output projection + MLP linears -- NOT OCAB's
own qkv projection, and NOT the window-attention (WindowAttention) qkv/proj at
all. AdaptSR's SwinIR ablation (cited in the M3b task) found the MLP and
overlapping-cross-attention paths drive domain-adaptation gain more than Q/K/V,
so Q/K/V stays frozen everywhere in this fine-tune.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a low-rank trainable update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear can only wrap nn.Linear, got {type(base)}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = r if alpha is None else alpha
        self.scaling = self.alpha / self.r

        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B left at zero: forward() below is exactly base(x) until trained.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + lora_out * self.scaling

    def extra_repr(self) -> str:
        return f"in_features={self.base.in_features}, out_features={self.base.out_features}, r={self.r}, alpha={self.alpha}"


def _wrap_linear(parent: nn.Module, attr: str, r: int, alpha: float | None, dropout: float) -> LoRALinear:
    linear = getattr(parent, attr)
    wrapped = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)
    setattr(parent, attr, wrapped)
    return wrapped


def inject_lora_into_hat(model, r: int = 8, alpha: float | None = None, dropout: float = 0.0) -> list[LoRALinear]:
    """Inserts LoRA adapters into every RHAG of a vendored HAT model, in place.

    Targets, per RHAG (model.layers[i].residual_group):
      - every HAB block's MLP: hab.mlp.fc1, hab.mlp.fc2
      - the RHAG's single OCAB module: overlap_attn.proj, overlap_attn.mlp.fc1,
        overlap_attn.mlp.fc2 -- explicitly not overlap_attn.qkv.

    Returns the list of inserted LoRALinear modules (for param counting /
    sanity checks). Does not touch requires_grad on the rest of the model --
    call freeze_backbone_except_lora(model) separately so the two concerns
    (where LoRA goes vs. what gets a gradient) stay independent and auditable.
    """
    inserted: list[LoRALinear] = []
    for rhag in model.layers:
        atten_blocks = rhag.residual_group
        for hab in atten_blocks.blocks:
            inserted.append(_wrap_linear(hab.mlp, "fc1", r, alpha, dropout))
            inserted.append(_wrap_linear(hab.mlp, "fc2", r, alpha, dropout))
        ocab = atten_blocks.overlap_attn
        inserted.append(_wrap_linear(ocab, "proj", r, alpha, dropout))
        inserted.append(_wrap_linear(ocab.mlp, "fc1", r, alpha, dropout))
        inserted.append(_wrap_linear(ocab.mlp, "fc2", r, alpha, dropout))
    return inserted


def freeze_backbone_except_lora(model) -> tuple[int, int]:
    """Sets requires_grad=False on every parameter except lora_A/lora_B.

    Returns (n_trainable_params, n_total_params) for reporting.
    """
    n_trainable = 0
    n_total = 0
    for name, p in model.named_parameters():
        n_total += p.numel()
        is_lora = name.endswith("lora_A") or name.endswith("lora_B")
        p.requires_grad = is_lora
        if is_lora:
            n_trainable += p.numel()
    return n_trainable, n_total


def lora_state_dict(model) -> dict[str, torch.Tensor]:
    """Just the LoRA adapter weights (lora_A/lora_B), for a small checkpoint."""
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()
            if name.endswith("lora_A") or name.endswith("lora_B")}
