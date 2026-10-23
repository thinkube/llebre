# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA on expert stacks, computed on the routed tokens, trains as PEFT's does."""

import copy

import pytest
import torch

from llebre import _moe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped matmuls run on a GPU")


@pytest.fixture
def moe():
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    torch.manual_seed(0)
    config = Qwen3_5MoeTextConfig(
        vocab_size=64, hidden_size=64, num_hidden_layers=2, layer_types=["full_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32, max_position_embeddings=64,
        initializer_range=0.2,
    )
    config._experts_implementation = "grouped_mm"
    model = Qwen3_5MoeForCausalLM(config).to("cuda", torch.bfloat16)
    peft_model = get_peft_model(model, LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0, target_modules=["q_proj", "v_proj"],
        target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]))
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(std=0.2)
    return peft_model


def run(model, ids):
    model.zero_grad()
    logits = model(input_ids=ids).logits.float()
    logits.pow(2).mean().backward()
    grads = {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach(), grads


def relative(x, reference):
    return ((x - reference).abs().max() / reference.abs().max().clamp(min=1e-6)).item()


def test_routed_lora_matches_peft(moe):
    routed = copy.deepcopy(moe)
    assert _moe.apply(routed) == 2
    ids = torch.randint(0, 64, (2, 24), device="cuda")
    out_peft, grads_peft = run(moe, ids)
    out_routed, grads_routed = run(routed, ids)
    assert relative(out_routed, out_peft) < 0.02
    # The adapters change the output far more than that tolerance: the comparison can tell
    with torch.no_grad(), moe.disable_adapter():
        assert relative(moe(input_ids=ids).logits.float(), out_peft) > 0.1
    expert_grads = [n for n in grads_peft if "experts" in n and "lora_" in n]
    assert expert_grads and grads_peft.keys() == grads_routed.keys()
    for name in grads_peft:
        assert relative(grads_routed[name], grads_peft[name]) < 0.05, name


def test_disabled_adapters_give_the_base_model(moe):
    routed = copy.deepcopy(moe)
    _moe.apply(routed)
    ids = torch.randint(0, 64, (1, 16), device="cuda")
    with torch.no_grad(), routed.disable_adapter(), moe.disable_adapter():
        assert relative(routed(input_ids=ids).logits.float(), moe(input_ids=ids).logits.float()) < 1e-3
