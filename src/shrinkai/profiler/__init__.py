"""Profiling: measuring and comparing a model's deployment footprint.

`Profiler.compare` (returning a `BenchmarkReport`) is the main entry point,
gathering parameter count, disk size, latency/throughput, optional accuracy,
and optional FLOPs (`count_flops`) for a teacher/student pair in one call,
also reachable as `Distiller.benchmark`, `Pruner.benchmark`,
`ChannelPruner.benchmark`, and `Quantizer.benchmark`. The individual measuring
functions (`measure_latency`, `count_parameters`, `estimate_model_size_mb`,
`get_process_ram_mb`, `get_device_memory_mb`, `count_flops`) are also usable
standalone.
"""

from .benchmark import BenchmarkReport, ModelProfile, Profiler
from .flops import count_flops
from .latency import measure_latency
from .memory import (
    count_parameters,
    estimate_model_size_mb,
    get_device_memory_mb,
    get_process_ram_mb,
)

__all__ = [
    "Profiler",
    "BenchmarkReport",
    "ModelProfile",
    "measure_latency",
    "count_parameters",
    "estimate_model_size_mb",
    "get_process_ram_mb",
    "get_device_memory_mb",
    "count_flops",
]
