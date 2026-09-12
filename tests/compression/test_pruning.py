import pytest
import torch
import torch.nn as nn

from shrinkai.compression.pruning import Pruner, PruningConfig

# ==========================================
# FIXTURES
# ==========================================


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)  # 100 weights
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)  # 50 weights

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


@pytest.fixture
def model():
    return DummyModel()


# ==========================================
# Pruning
# ==========================================


def test_pruningconfig_validation():
    """Test that invalid configurations raise appropriate errors."""
    with pytest.raises(ValueError, match="Invalid method"):
        PruningConfig(method="invalid_method")

    with pytest.raises(ValueError, match="Pruning amount must be between"):
        PruningConfig(amount=1.5)


def test_unstructured_pruning(model):
    """Test L1 unstructured pruning masks the exact percentage of weights."""
    amount = 0.3
    config = PruningConfig(method="unstructured", amount=amount, target_types=(nn.Linear,))
    pruner = Pruner(config)

    pruned_model = pruner.apply(model)

    # Verify weight_mask is applied to fc1
    assert hasattr(pruned_model.fc1, "weight_mask")

    # Check sparsity on fc1 (100 weights * 0.3 = 30 zeroed weights)
    zero_weights = (pruned_model.fc1.weight == 0).sum().item()
    assert zero_weights == 30, f"Expected 30 pruned weights, got {zero_weights}."


def test_structured_pruning(model):
    """Test L2 structured pruning drops entire neurons/channels."""
    amount = 0.4  # 40% of 10 output neurons = 4 neurons dropped
    config = PruningConfig(method="structured", amount=amount, target_types=(nn.Linear,))
    pruner = Pruner(config)

    pruned_model = pruner.apply(model)

    assert hasattr(pruned_model.fc1, "weight_mask")

    # Check that entire rows are zeroed out (dim 0)
    mask = pruned_model.fc1.weight_mask
    zeroed_rows = (mask.sum(dim=1) == 0).sum().item()
    assert zeroed_rows == 4, f"Expected 4 dropped neurons, got {zeroed_rows}."


def test_finalize_pruning(model):
    """Test that finalizing removes hooks and bakes the mask into the weights."""
    config = PruningConfig(method="unstructured", amount=0.5, target_types=(nn.Linear,))
    pruner = Pruner(config)

    pruned_model = pruner.apply(model)
    assert hasattr(pruned_model.fc1, "weight_mask")

    # Keep track of sparsity
    sparsity_before = (pruned_model.fc1.weight == 0).sum().item()

    final_model = pruner.finalize(pruned_model)

    # weight_mask attribute should be gone (PyTorch pruning hooks removed)
    assert getattr(final_model.fc1, "weight_mask", None) is None, (
        "Pruning mask hook was not removed."
    )

    # But weights should still have the zeros baked in
    sparsity_after = (final_model.fc1.weight == 0).sum().item()
    assert sparsity_before == sparsity_after, "Sparsity was lost during finalization."


def test_benchmark_does_not_crash_on_active_pruning_hooks(model):
    """Regression test: benchmark() must clone a model with live prune hooks.

    `copy.deepcopy` raises on modules with an active `torch.nn.utils.prune`
    reparametrization (masked weights are non-leaf tensors). benchmark() must
    not rely on it to produce a non-mutating finalized copy for evaluation.
    """
    config = PruningConfig(method="unstructured", amount=0.3, target_types=(nn.Linear,))
    pruner = Pruner(config)
    pruned_model = pruner.apply(model)

    report = pruner.benchmark(model, pruned_model, sample_input=torch.randn(1, 10))

    # The caller's pruned_model must remain untouched (hooks still attached).
    assert hasattr(pruned_model.fc1, "weight_mask")
    assert report.student.total_params == report.teacher.total_params
