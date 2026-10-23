<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/thinkube/llebre/main/assets/llebre-logo-dark.png">
    <img alt="llebre" src="https://raw.githubusercontent.com/thinkube/llebre/main/assets/llebre-logo-light.png" width="239" height="80">
  </picture>
</h1>

Fast LoRA and QLoRA fine-tuning on Hugging Face Transformers, PEFT and TRL,
built only from permissively licensed components, with the API of Unsloth's
`FastLanguageModel` and `FastModel`. Code written for Unsloth runs by changing
its import line. *Llebre* is Catalan for hare.

```python
import llebre as lb
from trl import SFTConfig, SFTTrainer

model, tokenizer = lb.FastLanguageModel.from_pretrained("org/model", max_seq_length=1024, load_in_4bit=True)
model = lb.FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)

trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset,
                     args=SFTConfig(output_dir="out", max_length=1024, per_device_train_batch_size=8))
trainer.train()
```

`from_pretrained` prints what the model will be trained with: the GPU, the
versions of PyTorch, Transformers and TRL, the model's weights, and whether
Liger and padding-free batches are on (and why not, when they are off).
`LLEBRE_QUIET=1` in the environment turns it off.

## Where the speed comes from

| Part | What it does |
|---|---|
| Tokenizer without legacy spaces | A SentencePiece tokenizer in legacy mode turns `<\|im_start\|>user` into `<\|im_start\|> user`; llebre loads it with `legacy=False`, so the model trains on the text its template writes |
| [Liger Kernel](https://github.com/linkedin/Liger-Kernel) (BSD-2) | Fused RMSNorm, RoPE and SwiGLU, and the output layer and loss computed in chunks without the full logits |
| Loss chunk budget | Liger's chunks sized 32 times larger than its default (`loss_chunk_budget`): on a large vocabulary the default makes a few dozen tokens per chunk and the GPU idles between thin matmuls |
| FlashAttention 2 and padding-free batches | The examples of a batch end to end, no compute on padding; TRL's `SFTTrainer` turns `padding_free` on for a llebre model |
| bitsandbytes NF4 (MIT) | QLoRA, without casting the norms to float32 |
| Example boundaries in linear-attention layers | Qwen3.5, Qwen3.6 and Qwen3-Next train padding-free with TRL 1.16 or later and [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT): llebre gives their short convolution the boundaries, which its PyTorch version ignores. With an older TRL, or without flash-linear-attention, these models train on padded batches |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)'s LoRA kernels (Apache-2.0), off by default | `get_peft_model(..., fused_lora=True)`: each adapted projection and each SwiGLU MLP computed with its adapter in one function with a hand-written backward pass. Models whose MLP is not SwiGLU are refused |

## Measured on one GPU

salamandraTA-2b-instruct (256K vocabulary) on one NVIDIA GB10, LoRA rank 16
on the seven projections, batch 8, at most 1,024 tokens, 8-bit AdamW, gradient
checkpointing, no_robots; tokens per second over steps 10 to 30, the same
tokens in every run:

| | LoRA | QLoRA | Peak memory, LoRA / QLoRA | Loss of the last 10 steps, LoRA / QLoRA |
|---|---|---|---|---|
| Unsloth 2026.10.7 | 2,072 | 2,003 | 5.5 / 3.8 GB | 2.492 / 2.561 |
| llebre 0.1.0 | 2,871 | 3,161 | 5.1 / 3.4 GB | 2.492 / 2.561 |
| llebre 0.1.0, `fused_lora=True` | 3,451 | 3,447 | 5.1 / 3.4 GB | 2.492 / 2.563 |

The gain is largest on models with a large vocabulary. Qwen3.5-4B (gated
DeltaNet layers), LoRA, the same settings: 508 tokens per second on padded
batches with TRL 0.29, 1,097 padding-free with TRL 1.16, the same loss.

Every table here comes from the scripts in [benchmarks](benchmarks), with the
recorded runs in [benchmarks/results](benchmarks/results).

## Mixture of experts

LoRA on an expert stack goes through PEFT's `target_parameters`:

```python
model = lb.FastModel.get_peft_model(model, r=16, lora_alpha=16, target_modules=[...],
                                    target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"])
```

PEFT turns the whole stack's weights into W + B·A for every expert at every
forward and backward; llebre computes each adapter's change on the tokens
routed to each expert only, two grouped matmuls of the LoRA rank's width,
and builds no full weight. Qwen3.6-35B-A3B on one NVIDIA GB10, LoRA rank 16 on
attention, gated DeltaNet, the shared expert and the 256 routed experts (941.9 M
parameters), 32 translation examples per step, padding-free:

| | Seconds per step | Steps per hour | Peak GPU memory |
|---|---|---|---|
| Unsloth (a training run of the same model, data and settings) | 10.5 | 343 | 76.0 GB |
| llebre, PEFT's expert LoRA | 15.6 | 231 | 76.5 GB |
| llebre | 8.1 | 443 | 74.9 GB |

## GRPO with vLLM

TRL's `GRPOTrainer` takes its settings from a llebre model as `SFTTrainer`
does, and generates through vLLM in the training process with
`GRPOConfig(use_vllm=True, vllm_mode="colocate")` (TRL 1.16 or later, vLLM
0.22 to 0.31 in the same environment). A model that Transformers loads under
other weight names than its checkpoint's, such as Qwen3.5's text model out of
its multimodal checkpoint, gets its weights to vLLM under the checkpoint's
names. A GRPO recipe on Qwen3.5-4B, 32 completions of up to 800 tokens per
step, one NVIDIA GB10:

| | First step | Each later step |
|---|---|---|
| Unsloth, without vLLM | 2,038 s | 122.8 s |
| llebre, without vLLM | 183 s | 161.7 s |
| llebre with vLLM | 149 s | 116.6 s |

## Several GPUs

- A model that fits on one GPU: DDP, one process per GPU, `torchrun
  --nproc_per_node N train.py`. Each process trains on its own GPU.
- A model that does not fit: FSDP2, `SFTConfig(..., **lb.fsdp_config(model))`
  with the model loaded by `from_pretrained(..., device_map="cpu")`, launched
  the same way; or its layers spread over the GPUs in one process,
  `from_pretrained(..., device_map="balanced")`, launched with `python`.

Under FSDP llebre keeps the model's BF16 weights in BF16 (`bf16` left unset:
Accelerate would turn them to FP32 before sharding) and refuses bitsandbytes'
optimizers, which cannot update sharded parameters (use
`optim="adamw_torch_fused"`). In a model spread over
GPUs, each part runs with its own GPU as the current device, so that Triton's
kernels reach their tensors between GPUs without peer-to-peer access.

Two NVIDIA RTX 3090 (24 GB) on PCIe, no NVLink, LoRA rank 16, padding-free,
no_robots:

| | Tokens per second | Memory per GPU |
|---|---|---|
| salamandraTA-7b, one GPU, 8 examples per step | 1,287 | 16.4 GB |
| salamandraTA-7b, DDP, 8 per GPU | 2,229 | 17.0 GB |
| salamandraTA-7b, FSDP2, 32 per GPU | 1,948 | 15.0 GB |
| salamandraTA-7b, layers spread, 8 per step | 1,289 | 6.8 / 10.4 GB |
| Qwen3-14B (29.5 GB, does not fit one GPU), layers spread, 8 per step | 641 | 14.3 / 16.9 GB |
| Qwen3-14B, FSDP2, 16 per GPU | 799 | 19.2 GB |
| Qwen3-14B QLoRA, FSDP2, 16 per GPU | 1,068 | 14.1 GB |
| Qwen3-14B QLoRA, DDP, 16 per GPU | 1,207 | 13.6 GB |

Between GPUs on PCIe, replicate the model when it fits each GPU (DDP), and
shard it only when it does not. For QLoRA under FSDP2, load with
`from_pretrained(..., load_in_4bit=True, quant_storage=torch.bfloat16)`, so that
each layer's packed 4-bit weights share the dtype FSDP2 shards.

## Large models on GPUs that share the system's memory

On an NVIDIA GB10 (DGX Spark) the GPU and the page cache share one memory,
and loading a checkpoint the usual way holds it twice. `from_pretrained`
reads each shard straight to such a GPU and drops it from the page cache: a
67 GB checkpoint loads in about a minute.

## Training on the answers only

```python
from llebre.chat_templates import train_on_responses_only

trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                  response_part="<|im_start|>assistant\n")
```

The loss counts only the tokens after each `response_part`, up to the next
`instruction_part`, in padded and padding-free batches alike.

## Saving

```python
model.save_pretrained_merged("out", tokenizer, save_method="merged_16bit")  # or "lora"
model.push_to_hub_merged("org/name", tokenizer, save_method="merged_16bit", token=...)
```

`merged_16bit` writes the base model with the adapter merged in, in the
model's 16-bit dtype; a QLoRA base is dequantized first. The model in memory
is left as it was.

## What it does not do yet

- Vision layers; GRPO with vLLM sharing one copy of the weights with the
  trainer; GGUF export; expert LoRA under FSDP2; training across
  several machines.

## Not affiliated with Unsloth

llebre reproduces the names and arguments of Unsloth's public API and contains
no Unsloth code. It includes Axolotl's LoRA kernels, under their Apache
License 2.0. Unsloth is a trademark of its owners. See [NOTICE](NOTICE).

## License

Apache License 2.0. Copyright Alejandro Martínez Corriá and the Thinkube contributors.
