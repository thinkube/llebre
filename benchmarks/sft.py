# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA and QLoRA SFT on one GPU: llebre, Unsloth, and Transformers + PEFT + TRL as they come.

    python sft.py --arm llebre --model BSC-LT/salamandraTA-2b-instruct --method lora

Every arm trains LoRA rank 16 (alpha 16, no dropout) on the seven projections
of each layer, with 8-bit AdamW, a constant learning rate of 1e-4, BF16 and
gradient checkpointing, on the same conversations in the same order. The
result is the tokens per second (padding not counted) from step --warmup to
the last, the peak GPU memory, and the mean loss of the first and last ten
steps, which tells whether the arms trained on the same tokens.
"""

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["llebre", "unsloth", "trl"], required=True)
parser.add_argument("--model", required=True, help="a Hugging Face model id or a local folder")
parser.add_argument("--method", choices=["lora", "qlora"], default="lora")
parser.add_argument("--fused", action="store_true", help="llebre: get_peft_model(fused_lora=True)")
parser.add_argument("--data", default="HuggingFaceH4/no_robots", help="a dataset id or a Parquet file of conversations")
parser.add_argument("--alternating", action="store_true", help="only conversations of strict user/assistant turns")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--results", default="results/sft.jsonl")
parser.add_argument("--progress", help="a file for a line per step")
a = parser.parse_args()

if a.arm == "unsloth":
    import unsloth  # noqa: F401  patches transformers and TRL before they load

import torch  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from common import Clock, chat_dataset, peak_gib, record  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
qlora = a.method == "qlora"
data = chat_dataset(a.data, a.batch * a.steps, alternating=a.alternating)
config = dict(output_dir="/tmp/llebre-sft", max_steps=a.steps, per_device_train_batch_size=a.batch,
              learning_rate=1e-4, lr_scheduler_type="constant", optim="adamw_8bit", bf16=True, logging_steps=1,
              save_strategy="no", report_to=[], seed=0, max_length=a.max_length,
              include_num_input_tokens_seen="non_padding", dataloader_num_workers=0)
peft_config = None

if a.arm == "llebre":
    import llebre as lb

    model, tokenizer = lb.FastLanguageModel.from_pretrained(a.model, max_seq_length=a.max_length, load_in_4bit=qlora)
    model = lb.FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=TARGETS,
                                                use_gradient_checkpointing=True, random_state=0, fused_lora=a.fused)
elif a.arm == "unsloth":
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(model_name=a.model, max_seq_length=a.max_length,
                                                         load_in_4bit=qlora, dtype=torch.bfloat16)
    model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16, lora_dropout=0.0, target_modules=TARGETS,
                                             use_gradient_checkpointing="unsloth", random_state=0)
    config["gradient_checkpointing"] = False  # Unsloth checkpoints itself
else:
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16) if qlora else None
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 quantization_config=quantization, device_map={"": 0})
    peft_config = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, target_modules=TARGETS, task_type="CAUSAL_LM")
    config["gradient_checkpointing"] = True
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

clock = Clock(a.progress)
trainer = SFTTrainer(model=model, args=SFTConfig(**config), train_dataset=data, processing_class=tokenizer,
                     peft_config=peft_config, callbacks=[clock])
torch.cuda.reset_peak_memory_stats()
trainer.train()
tokens_per_s, s_per_step = clock.rate(a.warmup, a.steps)
losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
record(a.results, {
    "benchmark": "sft", "arm": a.arm + ("-fused" if a.fused else ""), "model": a.model, "method": a.method,
    "batch": a.batch, "steps": a.steps, "tokens_per_s": round(tokens_per_s), "s_per_step": round(s_per_step, 3),
    "peak_gib": peak_gib(), "loss_first10": round(sum(losses[:10]) / 10, 4),
    "loss_last10": round(sum(losses[-10:]) / 10, 4), "padding_free": getattr(trainer.args, "padding_free", None),
})
