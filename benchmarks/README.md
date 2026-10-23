# Benchmarks

The measurements behind the tables in llebre's README, as scripts anyone can
run. Each script appends one line per run to a JSONL file in `results/`:
the settings, the numbers, the command, and the environment (GPUs, Python,
and the versions of PyTorch, Transformers, TRL, PEFT, Liger, bitsandbytes,
FlashAttention, flash-linear-attention, vLLM, Unsloth and llebre).

`results/recorded.jsonl` holds the measurements of the README, made on
2026-10-23 with the prototype scripts these were written from, with the same
settings. Every number carries its hardware and TRL version.

Requirements: llebre with its `qlora` extra, `datasets`, and for each arm what
it compares against (Unsloth in its own environment, vLLM for `--vllm`,
`python-constraint` for GRPO). `--progress FILE` writes a line per step.

## SFT on one GPU: `sft.py`

LoRA or QLoRA, rank 16 on the seven projections, 8-bit AdamW, 8 conversations
of no_robots per step, at most 1,024 tokens, gradient checkpointing; tokens
per second (padding not counted) from step 10 to step 30, peak memory, and the
loss of the first and last ten steps, the same in every arm when they train on
the same tokens.

```bash
python sft.py --arm llebre  --model BSC-LT/salamandraTA-2b-instruct --method lora
python sft.py --arm llebre  --model BSC-LT/salamandraTA-2b-instruct --method lora --fused
python sft.py --arm unsloth --model BSC-LT/salamandraTA-2b-instruct --method lora
python sft.py --arm trl     --model BSC-LT/salamandraTA-2b-instruct --method lora
```

`--arm trl` is Transformers, PEFT and TRL as they come: SDPA and padded batches.

## Mixture of experts: `moe.py`

LoRA rank 16 on Qwen3.6-35B-A3B's attention, gated DeltaNet, shared expert and
256 routed experts (941.9 M parameters), 32 examples per step, padding-free.

```bash
python moe.py --model Qwen/Qwen3.6-35B-A3B --data examples.jsonl   # {"prompt": ..., "answer": ...} per line
python moe.py --model Qwen/Qwen3.6-35B-A3B --data examples.jsonl --peft-experts
```

The recorded runs used a private translation corpus (short sentence pairs,
about 4,000 tokens per step of 32); without `--data` the script uses
no_robots, whose examples are longer, so its steps are not comparable to them.
The Unsloth figure is from a training of the same model, data and settings.

## GRPO: `grpo/grpo.py`

The zebra task: the model translates each clue of a zebra puzzle into a rule,
a constraint solver deduces the grid, and the reward is the share of correct
cells. Qwen3.5-4B, LoRA rank 16, 4 prompts of 8 completions of up to 800 tokens
per step.

```bash
cd grpo
python grpo.py --arm unsloth --model Qwen/Qwen3.5-4B
python grpo.py --arm llebre  --model Qwen/Qwen3.5-4B
torchrun --nproc_per_node 1 grpo.py --arm llebre --model Qwen/Qwen3.5-4B --vllm
```

`--vllm` needs vLLM and the training packages in one environment, and TRL 1.16
or later.

## Several GPUs: `multi_gpu.py`

LoRA or QLoRA rank 16, padding-free, no_robots conversations of strict
user/assistant turns.

```bash
torchrun --nproc_per_node 1 multi_gpu.py --mode ddp   --model BSC-LT/salamandraTA-7b-instruct
torchrun --nproc_per_node 2 multi_gpu.py --mode ddp   --model BSC-LT/salamandraTA-7b-instruct
torchrun --nproc_per_node 2 multi_gpu.py --mode fsdp  --model BSC-LT/salamandraTA-7b-instruct --batch 32
python                      multi_gpu.py --mode split --model BSC-LT/salamandraTA-7b-instruct
torchrun --nproc_per_node 2 multi_gpu.py --mode fsdp  --model Qwen/Qwen3-14B --batch 16
torchrun --nproc_per_node 2 multi_gpu.py --mode fsdp  --model Qwen/Qwen3-14B --batch 16 --qlora
torchrun --nproc_per_node 2 multi_gpu.py --mode ddp   --model Qwen/Qwen3-14B --batch 16 --qlora
python                      multi_gpu.py --mode split --model Qwen/Qwen3-14B
```

Recorded on two RTX 3090 (24 GB) on PCIe, without NVLink.
