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

## Where the speed comes from

| Part | What it does |
|---|---|
| Tokenizer without legacy spaces | A SentencePiece tokenizer in legacy mode turns `<\|im_start\|>user` into `<\|im_start\|> user`; llebre loads it with `legacy=False`, so the model trains on the text its template writes |
| [Liger Kernel](https://github.com/linkedin/Liger-Kernel) (BSD-2) | Fused RMSNorm, RoPE and SwiGLU, and the output layer and loss computed in chunks without the full logits |
| Loss chunk budget | Liger's chunks sized 32 times larger than its default (`loss_chunk_budget`): on a large vocabulary the default makes a few dozen tokens per chunk and the GPU idles between thin matmuls |
| FlashAttention 2 and padding-free batches | The examples of a batch end to end, no compute on padding; TRL's `SFTTrainer` turns `padding_free` on for a llebre model |
| bitsandbytes NF4 (MIT) | QLoRA, without casting the norms to float32 |
| [Axolotl](https://github.com/axolotl-ai-cloud/axolotl)'s LoRA kernels (Apache-2.0), off by default | `get_peft_model(..., fused_lora=True)`: each adapted projection and each SwiGLU MLP computed with its adapter in one function with a hand-written backward pass. Models whose MLP is not SwiGLU are refused |

## Measured

salamandraTA-2b-instruct (256K vocabulary) on one NVIDIA GB10, LoRA rank 16
on the seven projections, batch 8, at most 1,024 tokens, 8-bit AdamW, gradient
checkpointing, no_robots; tokens per second over steps 10 to 30, the same
tokens in every run:

| | LoRA | QLoRA | Peak memory, LoRA / QLoRA | Loss of the last 10 steps, LoRA / QLoRA |
|---|---|---|---|---|
| Unsloth 2026.10.7 | 2,072 | 2,003 | 5.5 / 3.8 GB | 2.492 / 2.561 |
| llebre 0.1.0 | 2,871 | 3,161 | 5.1 / 3.4 GB | 2.492 / 2.561 |
| llebre 0.1.0, `fused_lora=True` | 3,451 | 3,447 | 5.1 / 3.4 GB | 2.492 / 2.563 |

The gain is largest on models with a large vocabulary; it has not been
measured on larger models yet.

## What it does not do yet

- Padding-free batches on models with linear-attention layers (Qwen3.5,
  Qwen3-Next): this TRL does not pass them the example boundaries, so llebre
  trains them on padded batches.
- Vision layers, GRPO with vLLM sharing the weights, merged and GGUF saving,
  several GPUs.

## Not affiliated with Unsloth

llebre reproduces the names and arguments of Unsloth's public API and contains
no Unsloth code. It includes Axolotl's LoRA kernels, under their Apache
License 2.0. Unsloth is a trademark of its owners. See [NOTICE](NOTICE).

## License

Apache License 2.0. Copyright Alejandro Martínez Corriá and the Thinkube contributors.
