"""Binary-search GPU batch size benchmark.

Finds the maximum batch size that fits in GPU memory by running a
realistic conv1d workload at increasing sizes.  On CPU the benchmark
is skipped and a sensible default is returned.

Torch is imported inside function bodies to allow this module to be
imported for introspection even if torch is not yet available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def benchmark_max_batch_size(
    device: torch.device,
    sample_shape: tuple[int, ...] = (1, 48000),
    min_batch: int = 1,
    max_batch: int = 256,
) -> int:
    """Binary search for the maximum batch size that fits in GPU memory.

    Creates a random tensor of shape ``(batch, *sample_shape)`` on
    *device* and runs a ``conv1d`` operation (16 output channels,
    kernel size 1024, padding 512) to simulate realistic compute load.

    For CPU, the benchmark is skipped and a default of 32 is returned
    (CPU batch size is limited by system RAM, not GPU memory, and the
    binary search would be unreasonably slow).

    Args:
        device: Target device.
        sample_shape: Shape of a single sample.  Default is
            ``(1, 48000)`` representing 1 second of 48 kHz mono audio.
        min_batch: Lower bound for the search.
        max_batch: Upper bound for the search.

    Returns:
        The largest batch size that completed without OOM.
    """
    import torch

    from .memory import clear_gpu_memory

    if device.type == "cpu":
        return 32

    working_batch = min_batch
    low, high = min_batch, max_batch

    while low <= high:
        mid = (low + high) // 2
        oom_occurred = False

        try:
            x = torch.randn(mid, *sample_shape, device=device)
            # Conv1d expects (N, C_in, L) -- our x is (N, 1, 48000)
            weight = torch.randn(16, 1, 1024, device=device)
            _ = torch.nn.functional.conv1d(x, weight, padding=512)
            # If we reached here, this batch size works
            working_batch = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_occurred = True
                high = mid - 1
            else:
                raise
        finally:
            # Always clean up between iterations
            clear_gpu_memory(device)

        # Additional cleanup outside except when OOM occurred
        # (frame-reference-safe pattern from research pitfall #6)
        if oom_occurred:
            clear_gpu_memory(device)

    return working_batch


def format_benchmark_report(
    device: torch.device,
    max_batch_size: int,
    verbose: bool = False,
) -> str:
    """Format benchmark results for terminal display using rich markup.

    Args:
        device: The benchmarked device.
        max_batch_size: Result from :func:`benchmark_max_batch_size`.
        verbose: Include extra details (sample shape, methodology,
            post-benchmark memory state).

    Returns:
        A string with rich-markup formatting.
    """
    from .device import get_device_info

    info = get_device_info(device)
    device_name = info.get("name", str(device))

    lines: list[str] = []
    lines.append(
        f"Max batch size: [bold green]{max_batch_size}[/bold green] "
        f"(on {device_name})"
    )

    if verbose:
        lines.append("  Sample shape: (1, 48000) -- 1s of 48 kHz mono")
        lines.append("  Method: binary search with conv1d workload")

        from .memory import get_memory_info

        mem = get_memory_info(device)
        lines.append(
            f"  Memory after benchmark: "
            f"{mem['free_gb']:.1f} GB free / {mem['total_gb']:.1f} GB total"
        )

    return "\n".join(lines)


def run_benchmark(
    device: torch.device,
    verbose: bool = False,
) -> dict:
    """High-level benchmark runner.

    Runs :func:`benchmark_max_batch_size`, gathers device and memory
    context, prints a formatted report, and returns a dict suitable for
    storing in the project config.

    Args:
        device: Target device.
        verbose: Print verbose report.

    Returns:
        Dict with keys: ``max_batch_size``, ``device_type``,
        ``device_name``, ``memory_total_gb``.
    """
    from .device import get_device_info
    from .memory import get_memory_info

    max_batch = benchmark_max_batch_size(device)
    info = get_device_info(device)
    mem = get_memory_info(device)

    result: dict = {
        "max_batch_size": max_batch,
        "device_type": device.type,
        "device_name": info.get("name", str(device)),
        "memory_total_gb": mem.get("total_gb", 0.0),
    }

    # Print the formatted report
    report = format_benchmark_report(device, max_batch, verbose=verbose)
    try:
        from rich.console import Console

        console = Console()
        console.print(report)
    except ImportError:
        # Strip rich markup for plain output
        import re

        plain = re.sub(r"\[/?[^\]]+\]", "", report)
        print(plain)

    return result
