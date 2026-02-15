"""Memory querying and OOM-safe GPU operation wrapper.

Provides consistent memory info across CUDA, MPS, and CPU backends,
a ``clear_gpu_memory`` helper, and ``safe_gpu_operation`` which catches
OOM errors with proper exception-reference handling (recovery code
lives OUTSIDE the except block so frame references are released).

Torch is imported inside function bodies to allow this module to be
imported for introspection even if torch is not yet available.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import torch


def get_memory_info(device: torch.device) -> dict:
    """Get current memory state for *device*.

    Returns a dict with consistent keys:

    * ``allocated_gb`` -- memory currently used by PyTorch tensors.
    * ``total_gb``     -- total device/system memory.
    * ``free_gb``      -- memory available for new allocations.

    For CPU and MPS the system RAM figures come from *psutil*.
    """
    import torch

    info: dict = {
        "allocated_gb": 0.0,
        "total_gb": 0.0,
        "free_gb": 0.0,
    }

    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        info["allocated_gb"] = round(allocated / (1024**3), 4)
        info["total_gb"] = round(total / (1024**3), 2)
        info["free_gb"] = round(free / (1024**3), 2)

    elif device.type == "mps":
        import psutil

        mem = psutil.virtual_memory()
        mps_alloc = torch.mps.current_allocated_memory()
        info["allocated_gb"] = round(mps_alloc / (1024**3), 4)
        info["total_gb"] = round(mem.total / (1024**3), 2)
        info["free_gb"] = round(mem.available / (1024**3), 2)

    elif device.type == "cpu":
        import psutil

        mem = psutil.virtual_memory()
        info["total_gb"] = round(mem.total / (1024**3), 2)
        info["free_gb"] = round(mem.available / (1024**3), 2)

    return info


def clear_gpu_memory(device: torch.device) -> None:
    """Force GPU memory cleanup for *device*.

    Calls ``gc.collect()`` before and after the backend-specific cache
    clear.  No-op for CPU.
    """
    import torch

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    gc.collect()


def safe_gpu_operation(
    fn: Callable[..., Any],
    *args: Any,
    device: torch.device | None = None,
    **kwargs: Any,
) -> Any | None:
    """Run *fn* with OOM protection.

    If *fn* raises a CUDA/MPS out-of-memory ``RuntimeError``, memory is
    cleaned up and ``None`` is returned.  Non-OOM ``RuntimeError``
    exceptions are re-raised.

    **Critical:** Recovery code lives OUTSIDE the ``except`` block so
    that the Python frame reference held by the exception object is
    released before we attempt to free GPU memory (see research
    pitfall #6).

    Args:
        fn: Callable to execute.
        *args: Positional arguments forwarded to *fn*.
        device: Device used for cleanup.  If ``None``, cleanup is
            skipped (the operation may still succeed or re-raise).
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn*, or ``None`` on OOM.
    """
    oom_occurred = False
    result = None

    try:
        result = fn(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            oom_occurred = True
        else:
            raise

    # Recovery OUTSIDE except block -- frame references now released
    if oom_occurred:
        if device is not None:
            clear_gpu_memory(device)
        else:
            gc.collect()
        return None

    return result
