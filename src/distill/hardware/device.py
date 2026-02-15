"""Device detection, selection, smoke testing, and info reporting.

Provides the device abstraction layer for selecting MPS/CUDA/CPU,
validating the device works (smoke test), and querying device details.

Torch is imported inside function bodies to allow this module to be
imported for introspection even if torch is not yet available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def select_device(preference: str = "auto") -> torch.device:
    """Select compute device with validation.

    Args:
        preference: One of "auto", "mps", "cuda", "cuda:N", or "cpu".

    Returns:
        A validated ``torch.device`` ready for tensor operations.

    If *preference* is ``"auto"``, the best available device is detected
    automatically (CUDA -> MPS -> CPU).  For non-CPU devices a quick
    smoke test is run; on failure the function prints a warning via
    ``rich`` and falls back to CPU.
    """
    import torch

    if preference == "auto":
        device = _auto_detect()
    else:
        device = torch.device(preference)

    # Smoke-test non-CPU devices to catch "available but broken" GPUs.
    if device.type != "cpu":
        if not _smoke_test(device):
            try:
                from rich.console import Console

                console = Console(stderr=True)
                console.print(
                    f"[bold yellow]WARNING:[/bold yellow] {device} detected but "
                    "smoke test failed. Falling back to CPU."
                )
            except ImportError:
                print(
                    f"WARNING: {device} detected but smoke test failed. "
                    "Falling back to CPU."
                )
            device = torch.device("cpu")

    return device


def _auto_detect() -> torch.device:
    """Auto-detect best available device (CUDA -> MPS -> CPU).

    Uses the explicit backend checks rather than
    ``torch.accelerator.current_accelerator()`` for predictability.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _smoke_test(device: torch.device) -> bool:
    """Quick validation that *device* can perform basic tensor operations.

    Creates two 32x32 random tensors on the device, multiplies them,
    sums the result, and transfers it to CPU.  Returns ``False`` on any
    exception or if the result is NaN.
    """
    import torch

    try:
        a = torch.randn(32, 32, device=device)
        b = torch.randn(32, 32, device=device)
        c = torch.matmul(a, b)
        result = c.sum().item()  # force computation + transfer to CPU
        # NaN != NaN, so this returns False for NaN
        return result == result  # noqa: PLR0124
    except Exception:
        return False
    finally:
        _cleanup_cache(device)


def _cleanup_cache(device: torch.device) -> None:
    """Clear the GPU memory cache for *device* (no-op for CPU)."""
    import torch

    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def get_device_info(device: torch.device) -> dict:
    """Query detailed info about *device*.

    Returns a dict with at least the keys ``type``, ``name``, and
    ``memory_total_gb``.  Additional keys depend on the backend:

    * **CUDA**: ``memory_free_gb``, ``compute_capability``
    * **MPS**: ``memory_free_gb``, ``mps_allocated_gb``
    * **CPU**: ``cpu_count``
    """
    import platform

    import torch

    info: dict = {"type": device.type, "name": str(device)}

    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(device)
        free, total = torch.cuda.mem_get_info(device)
        info["memory_total_gb"] = round(total / (1024**3), 2)
        info["memory_free_gb"] = round(free / (1024**3), 2)
        info["compute_capability"] = torch.cuda.get_device_capability(device)

    elif device.type == "mps":
        import psutil

        info["name"] = "Apple Silicon (MPS)"
        mem = psutil.virtual_memory()
        info["memory_total_gb"] = round(mem.total / (1024**3), 2)
        info["memory_free_gb"] = round(mem.available / (1024**3), 2)
        info["mps_allocated_gb"] = round(
            torch.mps.current_allocated_memory() / (1024**3), 4
        )

    elif device.type == "cpu":
        import os

        import psutil

        mem = psutil.virtual_memory()
        info["memory_total_gb"] = round(mem.total / (1024**3), 2)
        # Attempt a human-readable CPU name
        cpu_name = platform.processor()
        if cpu_name:
            info["name"] = cpu_name
        else:
            info["name"] = f"CPU ({os.cpu_count()} cores)"
        info["cpu_count"] = os.cpu_count()

    return info


def format_device_report(
    device: torch.device, info: dict, verbose: bool = False
) -> str:
    """Format *info* for terminal display using rich markup.

    Args:
        device: The active ``torch.device``.
        info: Dict returned by :func:`get_device_info`.
        verbose: If ``True``, include extra details (PyTorch version,
            compute capability, CPU count, etc.).

    Returns:
        A string containing rich-markup formatted device information.
    """
    import torch

    lines: list[str] = []
    lines.append(f"[bold green]{info['name']}[/bold green]")

    mem_total = info.get("memory_total_gb")
    mem_free = info.get("memory_free_gb")
    if mem_total is not None:
        mem_str = f"  Memory: {mem_total:.1f} GB total"
        if mem_free is not None:
            mem_str += f", {mem_free:.1f} GB available"
        lines.append(mem_str)

    if verbose:
        lines.append(f"  PyTorch: {torch.__version__}")
        if device.type == "cuda":
            cap = info.get("compute_capability")
            if cap:
                lines.append(f"  Compute capability: {cap[0]}.{cap[1]}")
        elif device.type == "mps":
            mps_alloc = info.get("mps_allocated_gb", 0)
            lines.append(f"  MPS allocated: {mps_alloc:.4f} GB")
        elif device.type == "cpu":
            cpu_count = info.get("cpu_count")
            if cpu_count:
                lines.append(f"  CPU cores: {cpu_count}")

    return "\n".join(lines)
