from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from shrinkai.profiler.latency import measure_latency, synchronize_device


def test_synchronize_device_cpu():
    """Verifies that synchronizing a CPU device does not crash."""
    synchronize_device(torch.device("cpu"))


@patch("shrinkai.profiler.latency.time.perf_counter")
def test_measure_latency_math(mock_perf_counter):
    """Verifies the mathematical correctness of latency and FPS calculations."""
    model = nn.Linear(10, 5)
    sample_input = torch.randn(4, 10)

    mock_perf_counter.side_effect = [
        1.0,
        1.1,  # 100 ms
        2.0,
        2.1,  # 100 ms
    ]

    res = measure_latency(
        model, sample_input, device=torch.device("cpu"), num_runs=2, warmup_runs=1
    )

    assert res["batch_latency_ms"] == pytest.approx(100.0)
    # 100 ms / batch_size 4 = 25 ms
    assert res["sample_latency_ms"] == pytest.approx(25.0)
    # FPS = total samples 8 / total time 0.2s = 40 FPS
    assert res["fps"] == pytest.approx(40.0)
