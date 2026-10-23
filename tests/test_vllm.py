# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.conversion_mapping import get_model_conversion_mapping

from llebre._vllm import checkpoint_name


def test_a_text_model_out_of_a_multimodal_checkpoint_gets_its_checkpoint_names():
    config = Qwen3_5TextConfig(vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                               layer_types=["linear_attention", "full_attention"], num_attention_heads=4,
                               num_key_value_heads=2, head_dim=16, linear_key_head_dim=16, linear_value_head_dim=16,
                               linear_num_key_heads=2, linear_num_value_heads=4)
    model = Qwen3_5ForCausalLM(config)
    # What from_pretrained records when it loads the text model out of the multimodal checkpoint
    model._weight_conversions = get_model_conversion_mapping(model)
    assert checkpoint_name(model, "model.layers.0.linear_attn.in_proj_qkv.weight") == \
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    assert checkpoint_name(model, "model.embed_tokens.weight") == "model.language_model.embed_tokens.weight"
    assert checkpoint_name(model, "lm_head.weight") == "lm_head.weight"


def test_a_model_loaded_without_renamings_keeps_its_names(tiny_model):
    tiny_model._weight_conversions = None
    for name, _ in tiny_model.named_parameters():
        assert checkpoint_name(tiny_model, name) == name
