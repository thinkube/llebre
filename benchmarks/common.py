# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""What the benchmarks share: the data, a clock per step, and the result records."""

import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

NO_ROBOTS = "HuggingFaceH4/no_robots"


def chat_dataset(source: str, rows: int, alternating: bool = False):
    """Conversations as {"messages": [...]}: a Hugging Face dataset id or a Parquet file with a messages column.

    alternating: only conversations whose turns go user, assistant, user, ...,
    for chat templates that refuse any other order.
    """
    from datasets import load_dataset

    data = (load_dataset("parquet", data_files=source, split="train") if source.endswith(".parquet")
            else load_dataset(source, split="train"))
    if alternating:
        data = data.filter(lambda r: len(r["messages"]) % 2 == 0 and [m["role"] for m in r["messages"]]
                           == ["user", "assistant"] * (len(r["messages"]) // 2))
    if len(data) < rows:
        raise ValueError(f"{source} has {len(data)} usable rows; the run needs {rows}")
    data = data.select(range(rows))
    return data.map(lambda r: {"messages": [{"role": m["role"], "content": m["content"]} for m in r["messages"]]},
                    remove_columns=data.column_names)


def environment() -> dict:
    import importlib.metadata as md

    versions = {}
    for package in ("torch", "transformers", "trl", "peft", "liger-kernel", "bitsandbytes", "flash-attn",
                    "fla-core", "vllm", "unsloth", "llebre"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            pass
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {"gpus": gpus, "machine": platform.machine(), "python": platform.python_version(), "versions": versions}


def record(path: str, result: dict) -> None:
    """One result line, with the command and the environment, appended to a JSONL file."""
    line = {**result, "command": " ".join(sys.argv), "environment": environment(),
            "date": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    print("RESULT " + json.dumps(result), flush=True)
    if path and int(os.environ.get("RANK", 0)) == 0:
        with Path(path).open("a") as f:
            f.write(json.dumps(line) + "\n")


class Clock(TrainerCallback):
    """The time and the tokens seen at each step; a line per step to `progress`, if given."""

    def __init__(self, progress: str | None = None):
        self.marks, self.progress = {}, progress

    def on_train_begin(self, args, state, control, **kw):
        self.start = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        self.marks[state.global_step] = (now, state.num_input_tokens_seen)
        if self.progress and int(os.environ.get("RANK", 0)) == 0:
            took = now - self.marks.get(state.global_step - 1, (self.start, 0))[0]
            with open(self.progress, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} step {state.global_step}/{args.max_steps}: {took:.2f} s\n")

    def rate(self, first: int, last: int) -> tuple[float, float]:
        """Tokens per second and seconds per step from step `first` to step `last`."""
        (t0, n0), (t1, n1) = self.marks[first], self.marks[last]
        return (n1 - n0) / (t1 - t0), (t1 - t0) / (last - first)


def peak_gib() -> list:
    return [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())]
