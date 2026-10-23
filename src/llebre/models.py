# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""FastLanguageModel and FastModel: load a model for training, then add LoRA.

The names and arguments follow Unsloth's API, so a notebook moves between the
two by its import line. Every part underneath is Transformers, PEFT, TRL,
bitsandbytes and Liger Kernel.
"""

from typing import Any, Optional

import torch

from llebre import _fused, _liger

DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTENTION_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]
# Attention implementations that keep the examples of a padding-free batch apart
# through position ids
PADDING_FREE_ATTENTION = {"flash_attention_2", "flash_attention_3"}


def padding_free_ok(config, attn_implementation: str) -> bool:
    """Whether the model trains correctly on padding-free batches.

    It needs an attention that reads the example boundaries from the position
    ids, and no linear-attention layers: those (the gated DeltaNet layers of
    Qwen3.5 and Qwen3-Next) take their boundaries only as cu_seq_lens, which
    this TRL does not pass, so their examples would run into each other.
    """
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    layer_types = getattr(text, "layer_types", None) or []
    return attn_implementation in PADDING_FREE_ATTENTION and "linear_attention" not in layer_types


def load_tokenizer(model_name: str, **kwargs):
    """The tokenizer, without the legacy SentencePiece spaces.

    A SentencePiece tokenizer in legacy mode puts a space after every special
    token: "<|im_start|>user" becomes "<|im_start|> user", so a model would be
    trained on text it never saw. legacy=False tokenizes as written; a
    tokenizer that is not SentencePiece ignores the argument.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False, **kwargs)
    if tokenizer.pad_token is None:
        # Padding-free batches hold no padding; a padded batch masks its pad
        # positions out of the loss by the attention mask, so the end-of-sequence
        # token can stand as the pad token
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _dtype(dtype: Optional[torch.dtype]) -> torch.dtype:
    if dtype is not None:
        return dtype
    if not torch.cuda.is_available():
        raise RuntimeError("llebre trains on a CUDA GPU, and torch sees none")
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class FastLanguageModel:
    @staticmethod
    def from_pretrained(
        model_name: str,
        max_seq_length: int = 2048,
        dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        attn_implementation: str = "flash_attention_2",
        liger: bool = True,
        loss_chunk_budget: int = 32,
        device_map: Any = None,
        token: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        """The model and its tokenizer, ready for get_peft_model.

        dtype: bfloat16 where the GPU supports it, else float16, unless given.
        load_in_4bit: QLoRA, NF4 weights with double quantization, computed in dtype.
        liger: Liger's kernels and fused loss, which TRL's SFTTrainer applies;
            a model type Liger has no patches for is refused unless liger=False.
        loss_chunk_budget: Liger's chunk_mem_const for the fused loss (see llebre._liger).
        device_map: the whole model on GPU 0 unless given.
        """
        from transformers import AutoModelForCausalLM

        dtype = _dtype(dtype)
        hub = {"token": token, "trust_remote_code": trust_remote_code}
        quantization = None
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation=attn_implementation, quantization_config=quantization,
            device_map=device_map if device_map is not None else {"": 0}, **hub, **kwargs,
        )
        tokenizer = load_tokenizer(model_name, **hub)
        if liger:
            _liger.prepare(model, loss_chunk_budget)
        model.max_seq_length = max_seq_length
        model._llebre = {"padding_free": padding_free_ok(model.config, attn_implementation), "liger": liger}
        return model, tokenizer

    @staticmethod
    def get_peft_model(
        model,
        r: int = 16,
        target_modules: Optional[list] = None,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_gradient_checkpointing: Any = True,
        random_state: int = 3407,
        use_rslora: bool = False,
        modules_to_save: Optional[list] = None,
        fused_lora: bool = False,
        **kwargs,
    ):
        """LoRA adapters on the model, as a PEFT model.

        use_gradient_checkpointing: any true value recomputes each layer's
            activations in the backward pass (non-reentrant), saving memory for time.
        random_state: the seed of the adapters' initial weights.
        fused_lora: Axolotl's fused LoRA kernels on the adapted projections and
            SwiGLU MLPs (see llebre._fused); a model with another MLP activation
            is refused.
        Other keyword arguments go to PEFT's LoraConfig, for example
        target_parameters for the experts of a mixture-of-experts model.
        """
        from peft import LoraConfig, get_peft_model

        if getattr(model, "_llebre", None) is None:
            raise TypeError("get_peft_model takes a model loaded with llebre's from_pretrained")
        torch.manual_seed(random_state)
        config = LoraConfig(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias=bias,
                            target_modules=target_modules if target_modules is not None else DEFAULT_TARGETS,
                            use_rslora=use_rslora, modules_to_save=modules_to_save, task_type="CAUSAL_LM", **kwargs)
        # A 4-bit model is not passed through PEFT's prepare_model_for_kbit_training:
        # it turns the norms to float32, which costs memory and speed and changes
        # the dtype every layer computes in
        if use_gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        model.config.use_cache = False
        peft_model = get_peft_model(model, config)
        peft_model._llebre_gradient_checkpointing = bool(use_gradient_checkpointing)
        peft_model._llebre_fused_lora = fused_lora
        if fused_lora:
            _fused.apply(peft_model)
        return peft_model

    @staticmethod
    def for_inference(model):
        """Generation mode: no gradient checkpointing, the key-value cache on."""
        model.eval()
        if getattr(model, "is_gradient_checkpointing", False):
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        return model

    @staticmethod
    def for_training(model, use_gradient_checkpointing: Optional[bool] = None):
        """Training mode again, with gradient checkpointing as get_peft_model set it."""
        model.train()
        checkpointing = (getattr(model, "_llebre_gradient_checkpointing", False)
                         if use_gradient_checkpointing is None else use_gradient_checkpointing)
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        return model


class FastModel(FastLanguageModel):
    @staticmethod
    def get_peft_model(
        model,
        r: int = 16,
        finetune_language_layers: bool = True,
        finetune_attention_modules: bool = True,
        finetune_mlp_modules: bool = True,
        finetune_vision_layers: bool = False,
        target_modules: Optional[list] = None,
        **kwargs,
    ):
        """get_peft_model, with the targets chosen by kind of module.

        Vision layers are not trained by this version; finetune_vision_layers=True
        is refused.
        """
        if finetune_vision_layers:
            raise NotImplementedError("llebre does not train vision layers yet")
        if target_modules is None:
            if not finetune_language_layers:
                raise ValueError("nothing to train: finetune_language_layers is False and no target_modules")
            target_modules = (ATTENTION_TARGETS if finetune_attention_modules else []) + (
                MLP_TARGETS if finetune_mlp_modules else [])
            if not target_modules:
                raise ValueError("nothing to train: no attention and no MLP modules chosen")
        return FastLanguageModel.get_peft_model(model, r=r, target_modules=target_modules, **kwargs)
