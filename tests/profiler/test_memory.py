import torch.nn as nn

from shrinkai.profiler.memory import (
    count_parameters,
    estimate_model_size_mb,
    get_process_ram_mb,
)


def test_count_parameters():
    """Verifies that trainable and non-trainable parameters are accurately counted."""
    # 10 to 5 out -> 50 weights + 5 biases
    model = nn.Linear(10, 5)

    model.bias.requires_grad = False
    counts = count_parameters(model)

    assert counts["total_params"] == 55
    assert counts["trainable_params"] == 50
    assert counts["non_trainable_params"] == 5


def test_estimate_model_size_mb():
    """Verifies that the serialized model size estimation returns a positive float."""
    # approx 4MB
    model = nn.Linear(1000, 1000)

    size_mb = estimate_model_size_mb(model)

    assert isinstance(size_mb, float)
    assert size_mb > 0.0
    assert 3.5 < size_mb < 4.5


def test_get_process_ram_mb():
    """Verifies that RAM consumption returns a valid positive float."""
    ram_mb = get_process_ram_mb()
    assert isinstance(ram_mb, float)
    assert ram_mb > 0.0
