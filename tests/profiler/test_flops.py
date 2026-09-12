import logging

import torch
import torch.nn as nn

from shrinkai.profiler.flops import count_flops


def test_count_flops_matches_hand_computed_linear():
    """Verifies FLOPs for a single Linear layer match 2 * batch * in * out."""
    model = nn.Linear(10, 5, bias=False)
    sample_input = torch.randn(4, 10)

    flops = count_flops(model, sample_input, device="cpu")

    assert flops == 2 * 4 * 10 * 5


def test_count_flops_matches_hand_computed_conv2d():
    """Verifies FLOPs for a single Conv2d layer match the standard formula:
    2 * out_elements * (in_channels * kernel_h * kernel_w).
    """
    model = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    sample_input = torch.randn(1, 3, 16, 16)

    flops = count_flops(model, sample_input, device="cpu")

    expected = 2 * (16 * 16 * 8) * (3 * 3 * 3)
    assert flops == expected


def test_count_flops_restores_original_training_mode():
    """Verifies count_flops does not leave the model stuck in eval mode."""
    model = nn.Linear(4, 2)
    model.train()

    count_flops(model, torch.randn(1, 4), device="cpu")

    assert model.training


def test_count_flops_supports_multi_input_tuple():
    """Verifies a tuple of tensors is unpacked as separate forward arguments."""

    class TwoInputModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, a, b):
            return self.fc(a + b)

    model = TwoInputModel()
    flops = count_flops(model, (torch.randn(1, 4), torch.randn(1, 4)), device="cpu")

    assert flops == 2 * 1 * 4 * 2


def test_count_flops_warns_on_silent_undercount(caplog):
    """Verifies a warning is logged when a parameterized model reports 0 FLOPs
    (the signature of an unregistered/custom op, e.g. a quantized kernel).
    """

    class FakeCustomOpModel(nn.Module):
        """Has parameters, but its forward doesn't dispatch through them via a
        FLOPs-registered op — stands in for a custom/opaque kernel.
        """

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4))

        def forward(self, x):
            return x + self.weight.sum().detach()  # detach: no dispatch to count

    model = FakeCustomOpModel()

    with caplog.at_level(logging.WARNING):
        flops = count_flops(model, torch.randn(1, 4), device="cpu")

    assert flops == 0
    assert any("returned 0" in record.message for record in caplog.records)
