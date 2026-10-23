# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""TRL's SFTTrainer and GRPOTrainer, made to train a llebre model as llebre prepared it.

For a model from llebre's from_pretrained, the trainer takes these settings
from the model unless its config gives them:

- padding_free (SFT): the examples of a micro-batch laid end to end as one
  sequence, with position ids that restart at each example, so no compute
  goes to padding. from_pretrained decides whether the model can train that
  way (models.padding_free_ok).
- gradient_checkpointing: as get_peft_model's use_gradient_checkpointing set it.
- use_liger_kernel: as from_pretrained's liger set it. The trainer applies
  Liger's patches, and skips the metrics it would compute from the logits
  that Liger's fused loss never builds. For GRPO, TRL 0.29 computes the
  GRPO loss with Liger's chunked loss; TRL 1.x applies Liger's model kernels
  without a fused loss. With LoRA on parameters (target_parameters, for the
  experts of a mixture of experts), Liger's SwiGLU is left off: it would
  replace the forward of the module PEFT has wrapped.

One process training a model on one GPU uses that GPU only, never
nn.DataParallel, whatever the number of GPUs it sees.
With several processes (DDP), ddp_find_unused_parameters is False unless given.
Under FSDP, bf16 is False unless given (Accelerate would turn the BF16 weights
to FP32), and bitsandbytes' optimizers are refused (see _fsdp_settings).
A model spread over several GPUs gets its parts' devices again after the
trainer is built, which may have replaced the model's forward (see llebre._split).

After the trainer is built, a model with fused_lora gets its fused kernels
again, in place of the SwiGLU MLP forward that Liger set.

TRL 1.x refuses padding_free with a max_length and no packing, before its own
data preparation would cut the examples to that length. When llebre turns
padding-free on, the trainer is built without max_length, and the examples are
cut to it as they reach the data collator, keeping their start, as TRL 0.29
cuts them while preparing the dataset.
"""

import functools
import inspect
import os

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


# bitsandbytes' optimizers, which update plain tensors, not FSDP2's sharded ones
_BNB_OPTIMIZERS = {"adamw_8bit", "adamw_bnb_8bit", "paged_adamw_8bit", "paged_adamw_32bit", "paged_adamw",
                   "adam_8bit", "paged_lion_8bit", "lion_8bit", "paged_lion_32bit", "ademamix_8bit",
                   "paged_ademamix_8bit", "paged_ademamix_32bit", "rmsprop_bnb_8bit", "rmsprop_bnb"}


def _fsdp_settings(args, given) -> None:
    """A llebre model under FSDP: its BF16 weights stay BF16, and the optimizer can update shards."""
    optim = str(getattr(args.optim, "value", args.optim))
    if optim in _BNB_OPTIMIZERS:
        raise ValueError(f"optim={optim!r}: bitsandbytes' optimizers cannot update FSDP2's sharded parameters; "
                         "use optim='adamw_torch_fused' (LoRA's optimizer state is small)")
    if "bf16" not in given:
        # With mixed precision on, Accelerate turns every BF16 parameter to FP32
        # before sharding: twice the memory, and twice the weights the GPUs send
        # each other at every layer. The model is loaded in BF16 already.
        args.bf16 = False
    elif args.bf16:
        raise ValueError("bf16=True with FSDP: Accelerate then turns the model's BF16 weights to FP32 before "
                         "sharding, doubling their memory and the traffic between the GPUs; leave bf16 unset, "
                         "the model is loaded in BF16")


def _patch(trainer_cls, config_cls, kind: str, trl_1: bool) -> None:
    if getattr(trainer_cls.__init__, "_llebre", False):
        return

    config_init = config_cls.__init__

    @functools.wraps(config_init)
    def config_init_recording(self, *args, **kwargs):
        config_init(self, *args, **kwargs)
        object.__setattr__(self, _GIVEN, set(kwargs))

    config_cls.__init__ = config_init_recording

    trainer_init = trainer_cls.__init__
    signature = inspect.signature(trainer_init)

    @functools.wraps(trainer_init)
    def trainer_init_from_model(*call_args, **call_kwargs):
        bound = signature.bind(*call_args, **call_kwargs)
        model, args = bound.arguments.get("model"), bound.arguments.get("args")
        info = getattr(model, "_llebre", None)
        cut_to = None
        if info is not None:
            if args is None:
                name = str(model.config._name_or_path).rstrip("/").split("/")[-1]
                args = bound.arguments["args"] = config_cls(output_dir=f"{name}-{kind.upper()}")
            given = getattr(args, _GIVEN, set())
            if kind == "sft" and "padding_free" not in given:
                args.padding_free = info["padding_free"]
                if trl_1 and args.padding_free and not args.packing and args.max_length is not None:
                    cut_to, args.max_length = args.max_length, None
            if "use_liger_kernel" not in given:
                args.use_liger_kernel = info["liger"]
            if (args.use_liger_kernel and getattr(model, "_llebre_parameter_targets", False)
                    and "liger_kernel_config" not in given):
                # LoRA on parameters (the experts of a mixture of experts) wraps their
                # module; Liger's SwiGLU, which replaces that module's forward, would
                # run on PEFT's wrapper instead
                args.liger_kernel_config = {**(args.liger_kernel_config or {}), "swiglu": False}
            if getattr(args, "fsdp", None):
                _fsdp_settings(args, given)
            if int(os.environ.get("WORLD_SIZE", 1)) == 1 and not info.get("split") and getattr(args, "_n_gpu", 1) > 1:
                # One process, the model on one GPU, several GPUs visible: the Trainer
                # would wrap the model in nn.DataParallel, which TRL refuses
                args._n_gpu = 1
            if int(os.environ.get("WORLD_SIZE", 1)) > 1 and "ddp_find_unused_parameters" not in given:
                # Under DDP every adapter takes part in every step; looking for unused
                # parameters only costs a pass over the graph, and fails with
                # gradient checkpointing
                args.ddp_find_unused_parameters = False
            checkpointing = getattr(model, "_llebre_gradient_checkpointing", None)
            if checkpointing is not None and "gradient_checkpointing" not in given:
                args.gradient_checkpointing = checkpointing
                if checkpointing and "gradient_checkpointing_kwargs" not in given:
                    args.gradient_checkpointing_kwargs = {"use_reentrant": False}
            elif checkpointing and not args.gradient_checkpointing:
                # The config turns off what get_peft_model turned on
                model.gradient_checkpointing_disable()
        try:
            trainer_init(*bound.args, **bound.kwargs)
        finally:
            if cut_to is not None:
                args.max_length = cut_to
        trainer = bound.arguments["self"]
        if cut_to is not None:
            trainer.data_collator = CutToLength(trainer.data_collator, cut_to)
        if info is not None and info.get("split"):
            from llebre import _split

            base = trainer.model.get_base_model() if hasattr(trainer.model, "get_base_model") else trainer.model
            _split.apply(base)
            _split.count_on_output_device(trainer, base._llebre_output_device)
        if info is not None and getattr(model, "_llebre_fused_lora", False):
            from llebre import _fused

            _fused.apply(trainer.model)

    trainer_init_from_model._llebre = True
    trainer_cls.__init__ = trainer_init_from_model


def patch_trainers() -> None:
    import trl

    trl_1 = int(trl.__version__.split(".")[0]) >= 1
    _patch(trl.SFTTrainer, trl.SFTConfig, "sft", trl_1)
    _patch(trl.GRPOTrainer, trl.GRPOConfig, "grpo", trl_1)
