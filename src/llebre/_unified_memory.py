# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Loading a large checkpoint onto a GPU that shares the system's memory.

Transformers opens every safetensors shard with mmap and builds the GPU
tensors while the mapped pages are still in memory. On a GPU that shares the
system's memory (an NVIDIA GB10, a DGX Spark) the page cache and the GPU's
tensors come out of the same memory, so a checkpoint is held twice: a 67 GB
model needs some 134 GB and the load runs out of memory. While llebre loads a
model onto such a GPU, each shard is read straight to the GPU and its pages are
dropped from the page cache, so at most one shard is held twice.
"""

import contextlib
import os


def shares_system_memory(device: int = 0) -> bool:
    import torch

    return bool(torch.cuda.get_device_properties(device).is_integrated)


class _ShardOnGpu:
    """A safetensors shard read whole to the GPU, standing in for the mmap handle Transformers opens."""

    def __init__(self, safe_open, filename: str, device: str):
        with safe_open(filename, framework="pt", device=device) as f:
            self.tensors = {key: f.get_tensor(key) for key in f.keys()}
        fd = os.open(filename, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def keys(self):
        return self.tensors.keys()

    def get_slice(self, key):
        return self.tensors[key]

    def get_tensor(self, key):
        return self.tensors[key]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.tensors = {}


@contextlib.contextmanager
def shards_to_gpu(device: str = "cuda:0"):
    """While inside, Transformers' shard loading reads each shard to the GPU and drops its pages.

    Only the call that loads model shards (it passes a backend) is replaced;
    every other use of safe_open goes to safetensors.
    """
    from transformers import modeling_utils

    real = modeling_utils.safe_open

    def safe_open(filename, *args, **kwargs):
        if "backend" in kwargs:
            return _ShardOnGpu(real, filename, device)
        return real(filename, *args, **kwargs)

    modeling_utils.safe_open = safe_open
    try:
        yield
    finally:
        modeling_utils.safe_open = real
