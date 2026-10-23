# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""TRL's SFTTrainer, made to train a llebre model as llebre prepared it.

For a model from llebre's from_pretrained, the trainer takes three settings
from the model unless the SFTConfig gives them:

- padding_free: the examples of a micro-batch laid end to end as one
  sequence, with position ids that restart at each example, so no compute
  goes to padding. from_pretrained decides whether the model can train that
  way (models.padding_free_ok).
- gradient_checkpointing: as get_peft_model's use_gradient_checkpointing set it.
- use_liger_kernel: as from_pretrained's liger set it. The trainer applies
  Liger's patches, and skips the metrics it would compute from the logits
  that Liger's fused loss never builds.

After the trainer is built, a model with fused_lora gets its fused kernels
again, in place of the SwiGLU MLP forward that Liger set.

TRL 1.x refuses padding_free with a max_length and no packing, before its own
data preparation would cut the examples to that length. When llebre turns
padding-free on, the trainer is built without max_length, and the examples are
cut to it as they reach the data collator, keeping their start, as TRL 0.29
cuts them while preparing the dataset.
"""

import functools

_GIVEN = "_llebre_given"


def _cut(example: dict, max_length: int) -> dict:
    """Every per-token field of a tokenized example cut to its first max_length tokens."""
    length = len(example["input_ids"])
    if length <= max_length:
        return example
    return {key: value[:max_length]
            if not isinstance(value, (str, bytes, dict)) and hasattr(value, "__len__") and len(value) == length
            else value
            for key, value in example.items()}


class CutToLength:
    """A data collator that cuts each example to max_length before collating."""

    def __init__(self, collator, max_length: int):
        self.collator, self.max_length = collator, max_length

    def __call__(self, examples):
        return self.collator([_cut(example, self.max_length) for example in examples])


def patch_sft_trainer() -> None:
    import trl

    trainer_cls, config_cls = trl.SFTTrainer, trl.SFTConfig
    trl_1 = int(trl.__version__.split(".")[0]) >= 1
    if getattr(trainer_cls.__init__, "_llebre", False):
        return

    config_init = config_cls.__init__

    @functools.wraps(config_init)
    def config_init_recording(self, *args, **kwargs):
        config_init(self, *args, **kwargs)
        object.__setattr__(self, _GIVEN, set(kwargs))

    config_cls.__init__ = config_init_recording

    trainer_init = trainer_cls.__init__

    @functools.wraps(trainer_init)
    def trainer_init_from_model(self, model=None, args=None, *rest, **kwargs):
        info = getattr(model, "_llebre", None)
        cut_to = None
        if info is not None:
            if args is None:
                name = str(model.config._name_or_path).rstrip("/").split("/")[-1]
                args = config_cls(output_dir=f"{name}-SFT")
            given = getattr(args, _GIVEN, set())
            if hasattr(args, "padding_free") and "padding_free" not in given:
                args.padding_free = info["padding_free"]
                if trl_1 and args.padding_free and not args.packing and args.max_length is not None:
                    cut_to, args.max_length = args.max_length, None
            if "use_liger_kernel" not in given:
                args.use_liger_kernel = info["liger"]
            checkpointing = getattr(model, "_llebre_gradient_checkpointing", None)
            if checkpointing is not None and "gradient_checkpointing" not in given:
                args.gradient_checkpointing = checkpointing
                if checkpointing and "gradient_checkpointing_kwargs" not in given:
                    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
            elif checkpointing and not args.gradient_checkpointing:
                # The SFTConfig turns off what get_peft_model turned on
                model.gradient_checkpointing_disable()
        try:
            trainer_init(self, model, args, *rest, **kwargs)
        finally:
            if cut_to is not None:
                args.max_length = cut_to
        if cut_to is not None:
            self.data_collator = CutToLength(self.data_collator, cut_to)
        if info is not None and getattr(model, "_llebre_fused_lora", False):
            from llebre import _fused

            _fused.apply(self.model)

    trainer_init_from_model._llebre = True
    trainer_cls.__init__ = trainer_init_from_model
