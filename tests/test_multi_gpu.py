# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

from llebre import FastLanguageModel, fsdp_config


def test_fsdp_shards_by_the_models_decoder_layer(tiny_model):
    model = FastLanguageModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False)
    settings = fsdp_config(model)
    assert settings["fsdp"] == "full_shard auto_wrap"
    assert settings["fsdp_config"]["fsdp_version"] == 2
    assert settings["fsdp_config"]["transformer_layer_cls_to_wrap"] == ["LlamaDecoderLayer"]
