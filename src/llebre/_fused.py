# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Fused LoRA kernels on a PEFT model: get_peft_model(fused_lora=True).

The kernels (Axolotl's, in _vendor/axolotl) compute a projection and its LoRA
adapter in one autograd function with a hand-written backward pass, and an
MLP's gate, up and down projections with the SwiGLU activation as one.

They are attached to modules, not to source code: an MLP with LoRA on
gate_proj, up_proj and down_proj gets the fused SwiGLU MLP as its forward, and
every other LoRA projection on a linear layer gets the fused projection. A
model whose MLP is not SwiGLU is refused, since the MLP kernel computes SiLU.
"""

import types

from torch import nn

MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _kernels():
    try:
        from llebre._vendor.axolotl import lora
    except ImportError as e:
        raise ImportError(f"fused_lora needs bitsandbytes and Triton ({e}); install llebre[qlora]") from e
    return lora


def _on_linear(module) -> bool:
    """A LoRA layer over a linear layer, full precision or 4-bit."""
    from peft.tuners.lora import LoraLayer

    if not isinstance(module, LoraLayer):
        return False
    base = module.get_base_layer()
    if isinstance(base, nn.Linear):
        return True
    try:
        import bitsandbytes as bnb
    except ImportError:
        return False
    return isinstance(base, bnb.nn.Linear4bit)


def _fused_projection(kernels, module):
    original = module.forward

    def forward(x, *args, **kwargs):
        if args or kwargs or kernels._requires_projection_forward(module):
            return original(x, *args, **kwargs)
        return kernels.apply_lora_linear(module, x)

    return forward


def apply(model) -> dict:
    """Attach the kernels to the model's LoRA layers; returns how many of each.

    Applying again replaces the forwards set by another patch (Liger's SwiGLU
    MLP, which the trainer applies) and leaves the kernels' own in place.
    """
    kernels = _kernels()
    if len(model.active_adapters) != 1:
        raise NotImplementedError("fused_lora works with one active adapter")
    text = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    in_mlp = set()
    counts = {"mlp": 0, "projections": 0}
    for module in model.modules():
        projections = [getattr(module, name, None) for name in MLP_PROJECTIONS]
        if not all(p is not None and _on_linear(p) for p in projections):
            continue
        activation = getattr(text, "hidden_act", None)
        if activation != "silu":
            raise NotImplementedError(
                f"fused_lora fuses SwiGLU MLPs; {model.config.model_type} uses {activation!r}. "
                "Train it with fused_lora=False"
            )
        if getattr(module.forward, "__func__", None) is not kernels.apply_lora_mlp_swiglu:
            module.forward = types.MethodType(kernels.apply_lora_mlp_swiglu, module)
        in_mlp.update(id(p) for p in projections)
        counts["mlp"] += 1
    for module in model.modules():
        if id(module) in in_mlp or not _on_linear(module):
            continue
        if not getattr(module.forward, "_llebre_fused", False):
            module.forward = _fused_projection(kernels, module)
            module.forward._llebre_fused = True
        counts["projections"] += 1
    if not any(counts.values()):
        raise ValueError("fused_lora found no LoRA layer on a linear layer to fuse")
    return counts
