# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""LoRA on a mixture of experts, its routed experts included, with llebre.

    python moe.py --model Qwen/Qwen3.6-35B-A3B --data examples.jsonl

LoRA rank 16 (alpha 16) on full attention, the gated DeltaNet projections,
the shared expert and every routed expert (target_parameters), 32 examples
per step, padding-free, 8-bit AdamW; the loss on the answers only. --data is
a JSONL file of {"prompt": ..., "answer": ...}, each prompt a user turn with
thinking off; without it, the first user and assistant turns of no_robots.
--peft-experts trains the experts as PEFT computes them, every expert's full
weight at every pass, for comparison.
"""

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--data", help="a JSONL file of prompt and answer; no_robots without it")
parser.add_argument("--start", type=int, default=0, help="the first line of --data to use")
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--warmup", type=int, default=2)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--max-length", type=int, default=1024)
parser.add_argument("--peft-experts", action="store_true")
parser.add_argument("--results", default="results/moe.jsonl")
parser.add_argument("--progress")
a = parser.parse_args()

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

import llebre as lb  # noqa: E402
from common import Clock, chat_dataset, peak_gib, record  # noqa: E402

model, tokenizer = lb.LlebreModel.from_pretrained(a.model, max_seq_length=a.max_length, dtype=torch.bfloat16)
model = lb.LlebreModel.get_peft_model(
    model, r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
                    "gate_proj", "up_proj", "down_proj"],
    target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
    use_gradient_checkpointing=True, random_state=0)
if a.peft_experts:
    for module in model.modules():
        if hasattr(module, "_llebre_peft_forward"):
            module.forward = module._llebre_peft_forward
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

tok = getattr(tokenizer, "tokenizer", tokenizer)
head, tail = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False,
                                     add_generation_prompt=True, enable_thinking=False).split("\x00")
end = tok.eos_token if "<|im_end|>" not in tok.get_vocab() else "<|im_end|>"
if a.data:
    pairs = []
    with open(a.data) as f:
        for i, line in enumerate(f):
            if i >= a.start:
                ex = json.loads(line)
                pairs.append((ex["prompt"], ex["answer"]))
            if len(pairs) == a.batch * a.steps:
                break
else:
    pairs = [(r["messages"][0]["content"], r["messages"][1]["content"])
             for r in chat_dataset("HuggingFaceH4/no_robots", a.batch * a.steps, alternating=True)]
rows = []
for prompt, answer in pairs:
    p = tok(head + prompt + tail, add_special_tokens=False)["input_ids"]
    t = tok(answer + end, add_special_tokens=False)["input_ids"]
    rows.append({"input_ids": (p + t)[:a.max_length], "labels": ([-100] * len(p) + t)[:a.max_length]})

clock = Clock(a.progress)
args = SFTConfig(output_dir="/tmp/llebre-moe", per_device_train_batch_size=a.batch, max_steps=a.steps,
                 learning_rate=1e-4, optim="adamw_8bit", lr_scheduler_type="constant", bf16=True, logging_steps=1,
                 save_strategy="no", report_to=[], max_length=a.max_length, completion_only_loss=False,
                 loss_type="nll", remove_unused_columns=False, dataset_kwargs={"skip_prepare_dataset": True},
                 include_num_input_tokens_seen="non_padding", dataloader_num_workers=0, seed=0)
trainer = SFTTrainer(model=model, args=args, train_dataset=Dataset.from_list(rows), processing_class=tok,
                     callbacks=[clock])
torch.cuda.reset_peak_memory_stats()
trainer.train()
tokens_per_s, s_per_step = clock.rate(a.warmup, a.steps)
record(a.results, {
    "benchmark": "moe", "arm": "llebre-peft-experts" if a.peft_experts else "llebre", "model": a.model,
    "trainable_m": round(trainable / 1e6, 1), "batch": a.batch, "tokens_per_step": round(sum(len(r["input_ids"]) for r in rows) / a.steps),
    "s_per_step": round(s_per_step, 2), "steps_per_hour": round(3600 / s_per_step), "tokens_per_s": round(tokens_per_s),
    "peak_gib": peak_gib(), "losses": [round(h["loss"], 4) for h in trainer.state.log_history if "loss" in h],
})
