# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Under DDP (torchrun, 2 CPU processes on gloo) the trainer is set up for LoRA and trains."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_two_processes_train_with_unused_parameter_search_off(tmp_path):
    script = Path(__file__).parent / "ddp" / "train_two_processes.py"
    env = {**os.environ, "OUT": str(tmp_path), "CUDA_VISIBLE_DEVICES": ""}
    run = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(script)],
                         env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr[-3000:]
    results = [json.loads(line.split(" ", 1)[1]) for line in run.stdout.splitlines() if line.startswith("DDP-RESULT ")]
    assert sorted(r["rank"] for r in results) == [0, 1]
    assert all(r["world"] == 2 and r["find_unused"] is False and r["steps"] == 1 for r in results)
