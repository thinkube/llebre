# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import importlib

import pytest


@pytest.fixture
def fresh(monkeypatch):
    """llebre._liger with no patch applied, over a recording stand-in for Liger's chunked forward."""
    import liger_kernel.ops.fused_linear_cross_entropy as flce

    import llebre._liger as module

    calls = []
    monkeypatch.setattr(flce, "fused_linear_cross_entropy_forward", lambda *a, **kw: calls.append(kw))
    module = importlib.reload(module)
    return module, flce, calls


def test_the_budget_reaches_every_call(fresh):
    module, flce, calls = fresh
    module.set_loss_chunk_budget(32)
    flce.fused_linear_cross_entropy_forward(chunk_mem_const=1)
    module.set_loss_chunk_budget(8)
    flce.fused_linear_cross_entropy_forward()
    assert [c["chunk_mem_const"] for c in calls] == [32, 8]


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "32"])
def test_the_budget_is_a_positive_integer(fresh, bad):
    module, _, _ = fresh
    with pytest.raises(ValueError):
        module.set_loss_chunk_budget(bad)


def test_a_model_type_without_patches_is_refused(fresh, tiny_model):
    module, _, _ = fresh
    tiny_model.config.model_type = "no_such_model"
    with pytest.raises(NotImplementedError):
        module.prepare(tiny_model, 32)
