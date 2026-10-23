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
"""

import functools

_GIVEN = "_llebre_given"


def patch_sft_trainer() -> None:
    import trl

    trainer_cls, config_cls = trl.SFTTrainer, trl.SFTConfig
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
        if info is not None:
            if args is None:
                name = str(model.config._name_or_path).rstrip("/").split("/")[-1]
                args = config_cls(output_dir=f"{name}-SFT")
            given = getattr(args, _GIVEN, set())
            if hasattr(args, "padding_free") and "padding_free" not in given:
                args.padding_free = info["padding_free"]
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
        trainer_init(self, model, args, *rest, **kwargs)
        if info is not None and getattr(model, "_llebre_fused_lora", False):
            from llebre import _fused

            _fused.apply(self.model)

    trainer_init_from_model._llebre = True
    trainer_cls.__init__ = trainer_init_from_model
