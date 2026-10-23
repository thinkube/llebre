# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Saving a trained LoRA model: the adapter alone, or merged into 16-bit weights.

A merged save writes the base model with W + scaling * B @ A in place of each
adapted weight, computed in float32 and stored in the model's dtype. A 4-bit
base (QLoRA) is dequantized first, every 4-bit layer of it, so the result is
an ordinary 16-bit model. The model in memory is not changed: training or
generation can go on after the save.
"""

import copy
import tempfile
from pathlib import Path
from typing import Optional

import torch

SAVE_METHODS = ("lora", "merged_16bit")


def _dequantized(layer) -> Optional[torch.Tensor]:
    """The dense weight of a bitsandbytes 4-bit linear layer, or None for another layer."""
    try:
        import bitsandbytes as bnb
    except ImportError:
        return None
    if not isinstance(layer, bnb.nn.Linear4bit):
        return None
    return bnb.functional.dequantize_4bit(layer.weight.data, layer.weight.quant_state)


def _merged_weight(module, dtype) -> torch.Tensor:
    if len(module.active_adapters) != 1:
        raise NotImplementedError("a merged save takes one active adapter")
    adapter = module.active_adapters[0]
    if module.use_dora.get(adapter, False):
        raise NotImplementedError("a merged save of a DoRA adapter is not supported")
    base = module.get_base_layer()
    weight = _dequantized(base)
    if weight is None:
        if not isinstance(base, torch.nn.Linear):
            raise NotImplementedError(f"a merged save of LoRA on {type(base).__name__} is not supported")
        weight = base.weight
    a, b = module.lora_A[adapter].weight, module.lora_B[adapter].weight
    delta = (b.float() @ a.float()) * module.scaling[adapter]
    if getattr(module, "fan_in_fan_out", False):
        delta = delta.t()
    return (weight.float() + delta).to(dtype)


def merged_state_dict(peft_model) -> dict:
    """The base model's weights with the adapters merged and every 4-bit layer dequantized.

    They are stored in the dtype of the input embeddings, which are never
    quantized: the dtype the model was loaded in, or computes in when its other
    layers are 4-bit.
    """
    from peft.tuners.lora import LoraLayer

    base = peft_model.get_base_model()
    dtype = base.get_input_embeddings().weight.dtype
    state = {}
    done = []  # modules whose parameters are written as a whole: a LoRA layer, or a 4-bit layer
    for name, module in base.named_modules():
        if any(name.startswith(prefix + ".") for prefix in done):
            continue
        prefix = f"{name}." if name else ""
        if isinstance(module, LoraLayer):
            state[f"{prefix}weight"] = _merged_weight(module, dtype)
            bias = module.get_base_layer().bias
            if bias is not None:
                state[f"{prefix}bias"] = bias.detach().to(dtype)
            done.append(name)
            continue
        dense = _dequantized(module)
        if dense is not None:
            state[f"{prefix}weight"] = dense.to(dtype)
            if module.bias is not None:
                state[f"{prefix}bias"] = module.bias.detach().to(dtype)
            done.append(name)
            continue
        for pname, value in list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False)):
            if pname in getattr(module, "_non_persistent_buffers_set", set()):
                continue
            state[f"{prefix}{pname}"] = value.detach().to(dtype) if value.is_floating_point() else value.detach()
    return state


def save(peft_model, save_directory, tokenizer=None, save_method: str = "merged_16bit", **kwargs) -> None:
    if save_method not in SAVE_METHODS:
        raise ValueError(f"save_method must be one of {SAVE_METHODS}, not {save_method!r}; "
                         "a 4-bit merged save would lose precision and is not offered")
    directory = Path(save_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if save_method == "lora":
        peft_model.save_pretrained(directory, **kwargs)
    else:
        from huggingface_hub import save_torch_state_dict

        base = peft_model.get_base_model()
        state = merged_state_dict(peft_model)
        config = copy.deepcopy(base.config)
        if hasattr(config, "quantization_config"):
            del config.quantization_config
        config.dtype = next(v.dtype for v in state.values() if v.is_floating_point())
        config.save_pretrained(directory)
        if getattr(base, "generation_config", None) is not None:
            base.generation_config.save_pretrained(directory)
        save_torch_state_dict(state, directory, **kwargs)
    if tokenizer is not None:
        tokenizer.save_pretrained(directory)


def push(peft_model, repo_id: str, tokenizer=None, save_method: str = "merged_16bit", token: Optional[str] = None,
         private: Optional[bool] = None, commit_message: str = "Upload model") -> str:
    """Save as save() does, into a new or existing Hugging Face Hub repository; returns its URL."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    url = api.create_repo(repo_id, private=private, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        save(peft_model, directory, tokenizer=tokenizer, save_method=save_method)
        api.upload_folder(repo_id=repo_id, folder_path=directory, commit_message=commit_message)
    return str(url)
