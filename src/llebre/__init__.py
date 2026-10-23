# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Fast LoRA and QLoRA fine-tuning on Transformers, PEFT and TRL.

    import llebre as lb
    model, tokenizer = lb.FastLanguageModel.from_pretrained("org/model", load_in_4bit=True)
    model = lb.FastLanguageModel.get_peft_model(model, r=16)

Importing llebre makes TRL's SFTTrainer take padding-free batches and gradient
checkpointing from a llebre model, unless its SFTConfig sets them (see llebre._trl).
"""

from llebre import _trl
from llebre.models import FastLanguageModel, FastModel

__version__ = "0.1.0"
__all__ = ["FastLanguageModel", "FastModel", "__version__"]

_trl.patch_sft_trainer()
