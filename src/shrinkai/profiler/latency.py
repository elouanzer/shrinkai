import time

import torch
import torch.nn as nn


def synchronize_device(device: torch.device) -> None:
    """Synchronizes device streams to ensure precise time measurement.

    Args:
        device: Active PyTorch computing device.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def measure_latency(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    num_runs: int = 100,
    warmup_runs: int = 15,
) -> dict[str, float]:
    """Measures average inference latency and throughput (FPS / samples per second).

    Args:
        model: Model to profile.
        sample_input: Input batch tensor matching inference shape.
        device: Computing device.
        num_runs: Number of timed inference iterations.
        warmup_runs: Initial iterations discarded to warm up hardware caches.

    Returns:
        dict[str, float]: Latency per sample (ms), per batch (ms), and throughput (FPS).
    """
    was_training = model.training
    first_param = next(model.parameters(), None)
    original_device = first_param.device if first_param is not None else None

    model.eval()
    model.to(device)
    sample_input = sample_input.to(device)
    batch_size = sample_input.size(0)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(sample_input)
        synchronize_device(device)

    timings: list[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            synchronize_device(device)
            start_time = time.perf_counter()
            _ = model(sample_input)
            synchronize_device(device)
            timings.append(time.perf_counter() - start_time)

    avg_batch_latency_s = sum(timings) / len(timings)
    avg_batch_latency_ms = avg_batch_latency_s * 1000.0
    avg_sample_latency_ms = avg_batch_latency_ms / batch_size
    fps = (batch_size * num_runs) / sum(timings)

    model.train(mode=was_training)
    if original_device is not None:
        model.to(original_device)

    return {
        "batch_latency_ms": avg_batch_latency_ms,
        "sample_latency_ms": avg_sample_latency_ms,
        "fps": fps,
    }
