import io
import os

import psutil
import torch
import torch.nn as nn

from ..utils import resolve_device


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Counts total, trainable, and non-trainable parameters.

    Args:
        model: PyTorch model.

    Returns:
        dict[str, int]: Parameter count breakdown.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": total - trainable,
    }


def estimate_model_size_mb(model: nn.Module) -> float:
    """Estimates serialized state dictionary size on disk in Megabytes (MB).

    Args:
        model: PyTorch model.

    Returns:
        float: Estimated file size in MB.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / (1024 * 1024)
    return round(size_mb, 2)


def get_process_ram_mb() -> float:
    """Returns current host RAM consumption of the running Python process.

    Returns:
        float: Resident memory in MB.
    """
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024 * 1024), 2)


def get_device_memory_mb(device: torch.device | str = "auto") -> float:
    """Returns the currently allocated memory on the specified hardware accelerator.

    This is crucial for edge AI profiling, as VRAM is often the primary bottleneck.

    Args:
        device: Computing device ('auto', 'cuda', 'mps', 'cpu', or torch.device).

    Returns:
        float: Allocated accelerator memory in MB. Returns 0.0 for CPU
            (use `get_process_ram_mb` for host CPU RAM instead).
    """
    resolved_device = resolve_device(device)

    if resolved_device.type == "cuda":
        mem_bytes = torch.cuda.memory_allocated(resolved_device)
        return round(mem_bytes / (1024 * 1024), 2)

    elif resolved_device.type == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        mem_bytes = torch.mps.current_allocated_memory()
        return round(mem_bytes / (1024 * 1024), 2)

    return 0.0
