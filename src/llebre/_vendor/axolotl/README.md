# Vendored: Axolotl's LoRA kernels

The files here are from Axolotl (https://github.com/axolotl-ai-cloud/axolotl),
tag v0.21.0, commit 1f1eb05a391c43e65c1c5ee7ec6ea1b8a841d5d1, directory
src/axolotl/kernels/, under the Apache License 2.0 (LICENSE beside this file).
llebre uses them for get_peft_model(fused_lora=True).

| File | Changed |
|---|---|
| lora.py | no |
| quantize.py | no |
| utils.py | no |
| swiglu.py | the import of op_registry made relative |
| geglu.py | the import of op_registry made relative |
| op_registry.py | Python's logging in place of axolotl.utils.logging |

Each changed file says so in its first line.
