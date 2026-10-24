# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Fast LoRA and QLoRA fine-tuning on Transformers, PEFT and TRL.

    import llebre as lb
    model, tokenizer = lb.LlebreLanguageModel.from_pretrained("org/model", load_in_4bit=True)
    model = lb.LlebreLanguageModel.get_peft_model(model, r=16)

Importing llebre makes TRL's SFTTrainer and GRPOTrainer take their settings
from a llebre model, unless their config sets them (see llebre._trl), and
makes GRPO's weight updates reach vLLM under their checkpoint names (see llebre._vllm).
"""

from llebre import _trl, _vllm
from llebre.models import LlebreLanguageModel, LlebreModel
from llebre.multi_gpu import fsdp_config

__version__ = "0.1.0"
__all__ = ["LlebreLanguageModel", "LlebreModel", "fsdp_config", "__version__"]

_trl.patch_trainers()
_vllm.patch_vllm_generation()
