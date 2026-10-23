# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

WORDS = ["<unk>", "<pad>", "</s>", "the", "hare", "runs", "fast", "and", "sloth", "sleeps"]


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="</s>")


@pytest.fixture
def tiny_model():
    """A random two-layer Llama on the CPU, marked as llebre's from_pretrained marks it."""
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=len(WORDS), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64)
    model = LlamaForCausalLM(config)
    model._llebre = {"padding_free": True, "liger": False}
    return model
