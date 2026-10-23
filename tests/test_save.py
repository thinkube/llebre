# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import AutoModelForCausalLM

from llebre import FastLanguageModel

IDS = torch.tensor([[3, 4, 5, 6, 7, 8, 9]])


@pytest.fixture
def trained(tiny_model):
    model = FastLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    torch.manual_seed(1)
    for name, p in model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(std=0.1)
    return model.eval()


def test_the_merged_model_computes_what_the_adapted_one_does(trained, tokenizer, tmp_path):
    with torch.no_grad():
        expected = trained(input_ids=IDS).logits
    trained.save_pretrained_merged(tmp_path, tokenizer, save_method="merged_16bit")
    merged = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert not any("lora" in name for name, _ in merged.named_parameters())
    with torch.no_grad():
        torch.testing.assert_close(merged(input_ids=IDS).logits, expected, atol=1e-5, rtol=1e-5)
    assert (tmp_path / "tokenizer_config.json").exists()


def test_the_save_leaves_the_model_as_it_was(trained, tokenizer, tmp_path):
    with torch.no_grad():
        before = trained(input_ids=IDS).logits
    trained.save_pretrained_merged(tmp_path, tokenizer)
    with torch.no_grad():
        torch.testing.assert_close(trained(input_ids=IDS).logits, before)
    assert any("lora_A" in name for name, _ in trained.named_parameters())


def test_lora_saves_the_adapter_alone(trained, tmp_path):
    trained.save_pretrained_merged(tmp_path, save_method="lora")
    assert (tmp_path / "adapter_config.json").exists() and not (tmp_path / "config.json").exists()


@pytest.mark.parametrize("method", ["merged_4bit", "gguf"])
def test_other_save_methods_are_refused(trained, tmp_path, method):
    with pytest.raises(ValueError):
        trained.save_pretrained_merged(tmp_path, save_method=method)
