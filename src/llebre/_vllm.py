# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""GRPO generation through vLLM: the weights reach vLLM under their checkpoint names.

TRL starts vLLM on the model's files, so vLLM builds the architecture the
checkpoint names, and after each step TRL sends it the trained weights by the
names they have in the training model. The two can differ: Transformers loads
a text model out of a multimodal checkpoint (Qwen3.5) by renaming its weights
(model.language_model.X becomes model.X), and vLLM, holding the multimodal
architecture, knows only the checkpoint's names. Each name TRL sends is
renamed back as Transformers' own save_pretrained renames it, with the
renamings recorded on the model when it was loaded. A weight that a converter
(a merge or split of tensors) produced cannot be renamed alone, and is refused.

TRL 1.x keeps this in trl.generation.vllm_generation; llebre's vLLM support
needs TRL 1.16 or later.
"""

import functools

_PATCHED = "_llebre_checkpoint_names"


def _reverse_transforms(model):
    from transformers.core_model_loading import WeightConverter, WeightRenaming

    conversions = getattr(model, "_weight_conversions", None) or []
    reverse = [conversion.reverse_transform() for conversion in conversions[::-1]]
    return ([t for t in reverse if isinstance(t, WeightRenaming)],
            [t for t in reverse if isinstance(t, WeightConverter)])


def _base_model(model):
    model = getattr(model, "module", model)
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def checkpoint_name(model, name: str) -> str:
    """The name the weight has in the checkpoint the model was loaded from."""
    from transformers.core_model_loading import rename_source_key

    renamings, converters = _reverse_transforms(model)
    if not renamings and not converters:
        return name
    renamed, converter_pattern = rename_source_key(name, renamings, converters, reverse=True)
    if converter_pattern is not None:
        raise NotImplementedError(
            f"{name} was converted from several checkpoint tensors when the model was loaded "
            "(a merge or split); llebre cannot send it to vLLM by name"
        )
    return renamed


def patch_vllm_generation() -> None:
    try:
        from trl.generation.vllm_generation import VLLMGeneration
    except ImportError:
        return
    fix = VLLMGeneration._fix_param_name_to_vllm
    if getattr(fix, _PATCHED, False):
        return

    @functools.wraps(fix)
    def fix_to_checkpoint_name(self, name, extra_prefixes=None):
        name = fix(self, name, extra_prefixes=extra_prefixes)
        return checkpoint_name(_base_model(self.model), name)

    setattr(fix_to_checkpoint_name, _PATCHED, True)
    VLLMGeneration._fix_param_name_to_vllm = fix_to_checkpoint_name
