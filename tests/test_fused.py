# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest
import torch

from llebre import FastLanguageModel, FastModel

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the SwiGLU kernel runs on a GPU")


def with_adapters(model, **options):
    """The PEFT model with non-zero B matrices, so the adapters change the output."""
    peft_model = FastLanguageModel.get_peft_model(model, r=4, use_gradient_checkpointing=False, **options)
    torch.manual_seed(1)
    for name, p in peft_model.named_parameters():
        if "lora_B" in name:
            p.data.normal_(std=0.1)
    return peft_model


def outputs_and_gradients(model, ids):
    model.zero_grad()
    logits = model(input_ids=ids).logits
    logits.float().pow(2).mean().backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach(), grads


def assert_same_training(reference, fused, ids, tol):
    out_ref, grad_ref = outputs_and_gradients(reference, ids)
    out_fused, grad_fused = outputs_and_gradients(fused, ids)
    torch.testing.assert_close(out_fused, out_ref, atol=tol, rtol=tol)
    assert grad_ref.keys() == grad_fused.keys()
    for name in grad_ref:
        torch.testing.assert_close(grad_fused[name], grad_ref[name], atol=tol, rtol=tol, msg=name)


def test_fused_projections_train_like_peft(tiny_model):
    from llebre import _fused

    reference = with_adapters(copy.deepcopy(tiny_model), target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    fused = copy.deepcopy(reference)
    assert _fused.apply(fused) == {"mlp": 0, "projections": 8}
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9]])
    assert_same_training(reference, fused, ids, tol=1e-5)


def test_a_mlp_that_is_not_swiglu_is_refused(tiny_model):
    tiny_model.config.hidden_act = "gelu"
    with pytest.raises(NotImplementedError, match="gelu"):
        FastModel.get_peft_model(tiny_model, r=4, use_gradient_checkpointing=False, fused_lora=True)


def test_applying_again_keeps_the_kernels(tiny_model):
    from llebre import _fused

    model = with_adapters(tiny_model, target_modules=["q_proj", "o_proj"], fused_lora=True)
    q = model.base_model.model.model.layers[0].self_attn.q_proj
    forward = q.forward
    _fused.apply(model)
    assert q.forward is forward


@cuda
def test_fused_mlp_and_projections_train_like_peft(tiny_model):
    model = tiny_model.to("cuda", torch.bfloat16)
    reference = with_adapters(copy.deepcopy(model))
    fused = copy.deepcopy(reference)
    from llebre import _fused

    assert _fused.apply(fused) == {"mlp": 2, "projections": 8}
    ids = torch.tensor([[3, 4, 5, 6, 7, 8, 9]], device="cuda")
    assert_same_training(reference, fused, ids, tol=2e-2)
