import copy

import pytest
import torch
import torch.nn as nn

from shrinkai.compression.pruning import ChannelPruner
from shrinkai.compression.pruning.channel_pruner import _rank_keep_indices
from shrinkai.profiler.benchmark import BenchmarkReport

# ==========================================
# 1. FIXTURES / MOCK MODELS
# ==========================================


class SimpleCNN(nn.Module):
    """Conv -> BatchNorm -> ReLU -> Conv -> GlobalAvgPool -> Flatten -> Linear."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class MLP(nn.Module):
    """Linear -> BatchNorm1d -> ReLU -> Linear."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn1 = nn.BatchNorm1d(20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        return self.fc2(self.relu(self.bn1(self.fc1(x))))


@pytest.fixture
def cnn_sample():
    torch.manual_seed(0)
    return SimpleCNN().eval(), torch.randn(2, 3, 8, 8)


# ==========================================
# 2. CONFIG VALIDATION
# ==========================================


def test_channel_pruner_rejects_invalid_amount():
    """Verifies amount must be strictly between 0 and 1."""
    with pytest.raises(ValueError, match="amount must be between"):
        ChannelPruner(amount=0.0)
    with pytest.raises(ValueError, match="amount must be between"):
        ChannelPruner(amount=1.0)


# ==========================================
# 3. SUCCESSFUL PRUNING (SHAPES + FORWARD PASS)
# ==========================================


def test_prunes_conv_bn_conv_chain(cnn_sample):
    """Verifies conv1's output, bn1, and conv2's input all shrink consistently."""
    model, sample = cnn_sample

    pruned = ChannelPruner(amount=0.5).apply(model, target_layers=["conv1"], sample_input=sample)

    assert pruned.conv1.out_channels == 4
    assert pruned.bn1.num_features == 4
    assert pruned.conv2.in_channels == 4
    assert pruned.conv2.out_channels == 16  # untouched: not a target layer

    with torch.no_grad():
        out = pruned(sample)
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()


def test_prunes_through_global_avg_pool_into_linear(cnn_sample):
    """Verifies pruning conv2 (which feeds Linear via GAP+flatten) shrinks fc.in_features."""
    model, sample = cnn_sample

    pruned = ChannelPruner(amount=0.5).apply(model, target_layers=["conv2"], sample_input=sample)

    assert pruned.conv2.out_channels == 8
    assert pruned.fc.in_features == 8

    with torch.no_grad():
        out = pruned(sample)
    assert out.shape == (2, 4)


def test_prunes_multiple_layers_in_one_call(cnn_sample):
    """Verifies pruning both conv1 and conv2 together composes correctly."""
    model, sample = cnn_sample

    pruned = ChannelPruner(amount=0.5).apply(
        model, target_layers=["conv1", "conv2"], sample_input=sample
    )

    assert pruned.conv1.out_channels == 4
    assert pruned.conv2.in_channels == 4
    assert pruned.conv2.out_channels == 8
    assert pruned.fc.in_features == 8

    with torch.no_grad():
        out = pruned(sample)
    assert out.shape == (2, 4)


def test_prunes_mlp_chain():
    """Verifies ChannelPruner also works on plain Linear/BatchNorm1d chains."""
    model = MLP().eval()
    sample = torch.randn(4, 10)

    pruned = ChannelPruner(amount=0.5).apply(model, target_layers=["fc1"], sample_input=sample)

    assert pruned.fc1.out_features == 10
    assert pruned.bn1.num_features == 10
    assert pruned.fc2.in_features == 10

    with torch.no_grad():
        out = pruned(sample)
    assert out.shape == (4, 5)


def test_prunes_final_layer_with_no_downstream():
    """Verifies pruning the model's very last layer works (nothing to shrink after it)."""

    class LastLayerModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)

        def forward(self, x):
            return self.fc1(x)

    model = LastLayerModel().eval()
    sample = torch.randn(4, 10)

    pruned = ChannelPruner(amount=0.5).apply(model, target_layers=["fc1"], sample_input=sample)

    assert pruned.fc1.out_features == 10
    with torch.no_grad():
        out = pruned(sample)
    assert out.shape == (4, 10)


# ==========================================
# 4. NUMERICAL CORRECTNESS
# ==========================================


def test_pruned_output_matches_manually_masked_reference(cnn_sample):
    """Verifies physical pruning is mathematically equivalent to zeroing the same
    channels in the original (unshrunk) model — the defining correctness property
    of structured pruning.
    """
    model, sample = cnn_sample
    keep = _rank_keep_indices(model.conv1.weight, 0.5)

    pruned = ChannelPruner(amount=0.5).apply(
        copy.deepcopy(model), target_layers=["conv1"], sample_input=sample
    )

    masked = copy.deepcopy(model)
    with torch.no_grad():
        prune_mask = torch.ones(8, dtype=torch.bool)
        prune_mask[keep] = False
        masked.conv1.weight[prune_mask] = 0
        masked.conv1.bias[prune_mask] = 0
        masked.bn1.weight[prune_mask] = 0
        masked.bn1.bias[prune_mask] = 0
        masked.bn1.running_mean[prune_mask] = 0
        masked.bn1.running_var[prune_mask] = 1  # avoid div-by-zero; contributes 0 anyway

    with torch.no_grad():
        out_pruned = pruned(sample)
        out_masked = masked(sample)

    assert torch.allclose(out_pruned, out_masked, atol=1e-5)


def test_replacement_layers_preserve_eval_mode(cnn_sample):
    """Regression test: rebuilt BatchNorm must stay in eval mode, not default to
    train() (which would silently switch it to batch statistics instead of the
    copied running_mean/running_var).
    """
    model, sample = cnn_sample
    assert not model.training  # fixture already calls .eval()

    pruned = ChannelPruner(amount=0.5).apply(model, target_layers=["conv1"], sample_input=sample)

    assert not pruned.conv1.training
    assert not pruned.bn1.training


def test_replacement_layers_preserve_dtype(cnn_sample):
    """Regression test: rebuilt layers must keep the original model's dtype rather
    than silently defaulting to float32.
    """
    model, sample = cnn_sample
    model = model.double()
    sample = sample.double()

    pruned = ChannelPruner(amount=0.5).apply(model, target_layers=["conv1"], sample_input=sample)

    assert pruned.conv1.weight.dtype == torch.float64
    assert pruned.bn1.running_mean.dtype == torch.float64
    assert pruned.conv2.weight.dtype == torch.float64

    with torch.no_grad():
        out = pruned(sample)
    assert out.dtype == torch.float64


# ==========================================
# 5. REJECTED (UNSUPPORTED) TOPOLOGIES
# ==========================================


def test_rejects_unknown_layer_name(cnn_sample):
    model, sample = cnn_sample
    with pytest.raises(ValueError, match="not found in the model"):
        ChannelPruner(amount=0.3).apply(model, target_layers=["nope"], sample_input=sample)


def test_rejects_non_prunable_layer_type(cnn_sample):
    model, sample = cnn_sample
    with pytest.raises(ValueError, match="only supports Conv2d and Linear"):
        ChannelPruner(amount=0.3).apply(model, target_layers=["bn1"], sample_input=sample)


def test_rejects_grouped_convolution():
    class Grouped(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(8, 8, 3, padding=1, groups=8)
            self.conv2 = nn.Conv2d(8, 4, 1)

        def forward(self, x):
            return self.conv2(self.conv1(x))

    model = Grouped().eval()
    sample = torch.randn(1, 8, 4, 4)
    with pytest.raises(ValueError, match="grouped/depthwise"):
        ChannelPruner(amount=0.3).apply(model, target_layers=["conv1"], sample_input=sample)


def test_rejects_branching_topology():
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv2a = nn.Conv2d(8, 4, 3, padding=1)
            self.conv2b = nn.Conv2d(8, 4, 3, padding=1)

        def forward(self, x):
            x = self.conv1(x)
            return self.conv2a(x) + self.conv2b(x)

    model = Branch().eval()
    sample = torch.randn(1, 3, 4, 4)
    with pytest.raises(ValueError, match="Branching topologies"):
        ChannelPruner(amount=0.3).apply(model, target_layers=["conv1"], sample_input=sample)


def test_rejects_residual_connection():
    class Residual(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(8, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 8, 3, padding=1)

        def forward(self, x):
            return self.conv2(self.conv1(x)) + x

    model = Residual().eval()
    sample = torch.randn(1, 8, 4, 4)
    with pytest.raises(ValueError, match="unsupported call_function"):
        ChannelPruner(amount=0.3).apply(model, target_layers=["conv2"], sample_input=sample)


def test_rejects_unsafe_flatten_with_nontrivial_spatial_size():
    class UnsafeFlatten(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.fc = nn.Linear(8 * 4 * 4, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    model = UnsafeFlatten().eval()
    sample = torch.randn(1, 3, 4, 4)
    with pytest.raises(ValueError, match="not all 1"):
        ChannelPruner(amount=0.3).apply(model, target_layers=["conv1"], sample_input=sample)


# ==========================================
# 6. BENCHMARK
# ==========================================


def test_benchmark_returns_report(cnn_sample):
    model, sample = cnn_sample
    original = copy.deepcopy(model)

    pruned = ChannelPruner(amount=0.5).apply(model, target_layers=["conv1"], sample_input=sample)

    report = ChannelPruner(amount=0.5).benchmark(original, pruned, sample_input=sample)

    assert isinstance(report, BenchmarkReport)
    assert report.student.total_params < report.teacher.total_params


def test_benchmark_compute_flops_shows_real_reduction(cnn_sample):
    """Verifies compute_flops=True reports a genuine FLOPs reduction, since
    physical channel pruning (unlike mask-based Pruner) actually shrinks the
    computation graph.
    """
    model, sample = cnn_sample
    original = copy.deepcopy(model)

    pruned = ChannelPruner(amount=0.5).apply(model, target_layers=["conv1"], sample_input=sample)

    report = ChannelPruner(amount=0.5).benchmark(
        original, pruned, sample_input=sample, compute_flops=True
    )

    assert report.teacher.flops is not None
    assert report.student.flops is not None
    assert report.student.flops < report.teacher.flops
