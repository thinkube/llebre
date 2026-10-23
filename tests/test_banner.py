# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

import transformers
import trl

from llebre import _banner


def test_the_banner_says_what_the_model_trains_with():
    text = _banner.text("0.1.0", model_type="llama", weights="4-bit NF4 computing in bfloat16", liger=True,
                        padding_free=False, padding_free_reason="attention is sdpa")
    lines = text.splitlines()
    assert len(lines) == 4
    assert "llebre 0.1.0" in lines[0]
    assert f"Transformers {transformers.__version__}" in lines[1] and f"TRL {trl.__version__}" in lines[1]
    assert "Model: llama, 4-bit NF4 computing in bfloat16. Liger: on. Padding-free: off (attention is sdpa)" in lines[2]
    # The text starts in the same column on every line, beside the hare
    assert len({line.index(part) for line, part in zip(lines, ["llebre", "GPU", "Model", "Apache"])}) == 1


def test_llebre_quiet_turns_it_off(monkeypatch, capsys):
    monkeypatch.setenv("LLEBRE_QUIET", "1")
    _banner.show(model_type="llama", weights="bfloat16", liger=True, padding_free=True, padding_free_reason=None)
    assert capsys.readouterr().out == ""
    monkeypatch.delenv("LLEBRE_QUIET")
    _banner.show(model_type="llama", weights="bfloat16", liger=True, padding_free=True, padding_free_reason=None)
    assert "Padding-free: on" in capsys.readouterr().out
