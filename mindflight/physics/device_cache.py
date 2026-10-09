"""Keep small constant tensors (motor layout, gravity, inertia) on the GPU, made once and reused every step.

Why bother? `step` used to build these from Python numbers or CPU tensors every call, e.g.
`state.new_tensor([0, 0, -9.81])`. On a GPU each of those is a CPU -> GPU copy. One copy is cheap, but:
- the CPU has to stop and hand the data over, every physics step, and
- a CUDA graph (see `fast.py`) can't record a copy from ordinary CPU memory, so the step couldn't be captured.

`@cached_on_device` fixes both: the first call for a given (device, dtype, ...) builds the tensor, later calls get
the same tensor back. The numbers are exactly the ones the old code made. The tensors are shared by every caller,
so treat them as read-only.
"""

import functools

import torch


def cached_on_device(build):
    """Decorator for `build(device, dtype, *args) -> tensor (or tuple of tensors)`: run it once per argument set.

    Every argument must be hashable (torch.device, torch.dtype, and the frozen `CrazyflieParams` all are).

    Two details:
    - The tensors are made with inference mode switched off, so they are normal tensors even if the first call
      happens inside `torch.inference_mode()` (inference tensors can't be used later by code that records autograd).
    - Inside `torch.compile`, we skip the cache and call `build` directly: the compiler bakes the numbers into its
      own kernels, so there is nothing to cache.
    """
    cache = {}  # (device, dtype, *args) -> what build returned

    @functools.wraps(build)
    def get(device: torch.device, dtype: torch.dtype, *args):
        if torch.compiler.is_compiling():
            return build(device, dtype, *args)
        key = (device, dtype, *args)
        value = cache.get(key)
        if value is None:  # first call for this device/dtype/params: build it once
            with torch.inference_mode(False):
                value = cache[key] = build(device, dtype, *args)
        return value

    get.cache_clear = cache.clear
    return get
