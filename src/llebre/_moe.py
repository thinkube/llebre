# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA on the experts of a mixture of experts, computed on the routed tokens only.

PEFT puts LoRA on an expert stack (target_parameters) by registering, at each
forward, a parametrization that turns the stack's weight into W + scale * lhs @ rhs
for all experts. On Qwen3.6-35B-A3B that builds and copies about 1 GB of weights
per layer at every forward and backward, though a token reaches 8 of the 256
experts: half the GPU time of a training step.

Here the outermost of PEFT's wrappers around the experts runs the experts
itself, as Transformers' grouped_mm implementation does, on the base weights
unchanged, and adds each adapter's change to the tokens routed to each expert:
x @ W^T + scale * (x @ rhs^T) @ lhs^T, two grouped matmuls whose inner size is
the LoRA rank. Gradients reach the adapters through those products. The
forward is the same function of the inputs and the weights; only the
intermediate full weights are not built.
"""

import types

import torch

_PATCHED = "_llebre_routed_lora"


def _wrapper_chain(outer):
    """PEFT's ParamWrappers around an experts module, outermost first, and the module."""
    from peft.tuners.lora.layer import ParamWrapper

    chain, module = [], outer
    while isinstance(module, ParamWrapper):
        chain.append(module)
        module = module.base_layer
    return chain, module


def _grouped_lora(x, lhs, rhs, scaling, offsets, is_transposed):
    """scale * x @ delta per group, delta = lhs @ rhs in the layout of the experts' weight."""
    from transformers.integrations.moe import _grouped_mm

    # The grouped matmul needs every row stride to be a multiple of 16 bytes; the
    # rank dimension is padded with zeros to that, which leaves lhs @ rhs unchanged
    pad = -lhs.shape[-1] % (16 // lhs.element_size())
    if pad:
        lhs = torch.nn.functional.pad(lhs, (0, pad))
        rhs = torch.nn.functional.pad(rhs, (0, 0, 0, pad))
    if is_transposed:  # weight (experts, in, out): x @ delta
        low = _grouped_mm(x, lhs.contiguous(), offs=offsets)
        return _grouped_mm(low, rhs.contiguous(), offs=offsets) * scaling
    # weight (experts, out, in): x @ delta^T
    low = _grouped_mm(x, rhs.transpose(-2, -1).contiguous(), offs=offsets)
    return _grouped_mm(low, lhs.transpose(-2, -1).contiguous(), offs=offsets) * scaling


def _routed_forward(experts, hidden_states, top_k_index, top_k_weights, factors):
    """Transformers' grouped_mm experts forward (5.17), with the adapters' change added per projection."""
    from transformers.integrations.moe import _grouped_linear

    device = hidden_states.device
    num_top_k = top_k_index.size(-1)
    num_tokens, hidden_dim = hidden_states.size(0), hidden_states.size(-1)
    sample_weights = top_k_weights.reshape(-1)
    expert_ids = top_k_index.reshape(-1)
    expert_ids_g, perm = torch.sort(expert_ids)
    selected = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]
    histc_input = expert_ids_g.float() if device.type in ("cpu", "mps") else expert_ids_g.int()
    tokens_per_expert = torch.histc(histc_input, bins=experts.num_experts, min=0, max=experts.num_experts - 1)
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)
    sentinel = (expert_ids_g >= experts.num_experts).unsqueeze(-1)
    expert_ids_g.clamp_(max=experts.num_experts - 1)
    selected = selected.masked_fill(sentinel, 0.0)

    def project(x, name, bias_name):
        weight = getattr(experts, name)
        bias = getattr(experts, bias_name)[expert_ids_g] if experts.has_bias else None
        out = _grouped_linear(x, weight, offsets, bias=bias, is_transposed=experts.is_transposed)
        if name in factors:
            out = out + _grouped_lora(x.to(weight.dtype), *factors[name], offsets, experts.is_transposed)
        return out.masked_fill(sentinel, 0.0)

    if experts.has_gate:
        out = experts._apply_gate(project(selected, "gate_up_proj", "gate_up_proj_bias"))
    else:
        out = experts.act_fn(project(selected, "up_proj", "up_proj_bias"))
    out = project(out, "down_proj", "down_proj_bias")
    weighted = out * sample_weights_g.unsqueeze(-1)
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(perm.size(0), device=device)
    return weighted[inverse].view(num_tokens, num_top_k, hidden_dim).sum(dim=1).to(hidden_states.dtype)


def _forward(self, hidden_states, top_k_index, top_k_weights, *args, **kwargs):
    chain, experts = _wrapper_chain(self)
    adapters = self.active_adapters
    if (args or kwargs or self.disable_adapters or any(w.merged for w in chain) or len(adapters) != 1
            or any(adapters[0] not in w.lora_A for w in chain)):
        return self._llebre_peft_forward(hidden_states, top_k_index, top_k_weights, *args, **kwargs)
    factors = {w.parameter_name: w.get_delta_factors(adapters[0]) for w in chain}
    return _routed_forward(experts, hidden_states, top_k_index, top_k_weights, factors)


setattr(_forward, _PATCHED, True)


def apply(model) -> int:
    """The routed LoRA on every expert stack PEFT has wrapped; returns how many."""
    from peft.tuners.lora.layer import ParamWrapper

    patched = 0
    for name, module in model.named_modules():
        if not isinstance(module, ParamWrapper) or name.endswith(".base_layer"):
            continue  # an inner wrapper: its outermost one runs the chain
        chain, experts = _wrapper_chain(module)
        if not all(hasattr(experts, a) for a in ("num_experts", "has_gate", "has_bias", "is_transposed")):
            raise NotImplementedError(
                f"{name}: LoRA on parameters of {type(experts).__name__}, which is not a Transformers experts module")
        for w in chain:
            for dropout in w.lora_dropout.values():
                if not isinstance(dropout, torch.nn.Identity):
                    raise NotImplementedError(f"{name}: LoRA dropout on expert parameters is not supported")
        if not getattr(module.forward, _PATCHED, False):
            module._llebre_peft_forward = module.forward
            module.forward = types.MethodType(_forward, module)
        patched += 1
    return patched
