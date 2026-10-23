# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import LlamaConfig, PretrainedConfig

from llebre import FastLanguageModel, FastModel
from llebre.models import ATTENTION_TARGETS, DEFAULT_TARGETS, padding_free_ok


def test_padding_free_needs_flash_attention():
    config = LlamaConfig()
    assert padding_free_ok(config, "flash_attention_2")
    assert padding_free_ok(config, "flash_attention_3")
    assert not padding_free_ok(config, "sdpa")
    assert not padding_free_ok(config, "eager")


def test_padding_free_refused_with_linear_attention_layers():
    config = PretrainedConfig(layer_types=["linear_attention", "linear_attention", "full_attention"])
    assert not padding_free_ok(config, "flash_attention_2")


def lora_names(model):
    return {name.split(".lora_")[0].split(".")[-1] for name, _ in model.named_parameters() if ".lora_" in name}


def test_get_peft_model_trains_only_the_adapters(tiny_model):
    model = FastLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_" in name for name in trainable)
    assert lora_names(model) == set(DEFAULT_TARGETS)
    assert model._llebre == {"padding_free": True, "liger": False}


def test_random_state_fixes_the_initial_adapters(tiny_model):
    a = FastLanguageModel.get_peft_model(tiny_model, r=4, random_state=7, use_gradient_checkpointing=False)
    first = {n: p.detach().clone() for n, p in a.named_parameters() if "lora_A" in n}
    a.unload()
    b = FastLanguageModel.get_peft_model(tiny_model, r=4, random_state=7, use_gradient_checkpointing=False)
    for name, p in b.named_parameters():
        if "lora_A" in name:
            assert torch.equal(p, first[name])


def test_gradient_checkpointing_follows_the_argument(tiny_model):
    model = FastLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=True)
    assert model.base_model.model.is_gradient_checkpointing
    FastLanguageModel.for_inference(model)
    assert not model.base_model.model.is_gradient_checkpointing and model.config.use_cache
    FastLanguageModel.for_training(model)
    assert model.base_model.model.is_gradient_checkpointing and not model.config.use_cache


def test_get_peft_model_refuses_a_model_llebre_did_not_load(tiny_model):
    del tiny_model._llebre
    with pytest.raises(TypeError):
        FastLanguageModel.get_peft_model(tiny_model)


def test_fast_model_chooses_targets_by_kind(tiny_model):
    model = FastModel.get_peft_model(tiny_model, r=4, finetune_mlp_modules=False, use_gradient_checkpointing=False)
    assert lora_names(model) == set(ATTENTION_TARGETS)


def test_fast_model_refuses_vision_layers_and_empty_choices(tiny_model):
    with pytest.raises(NotImplementedError):
        FastModel.get_peft_model(tiny_model, finetune_vision_layers=True)
    with pytest.raises(ValueError):
        FastModel.get_peft_model(tiny_model, finetune_attention_modules=False, finetune_mlp_modules=False)
