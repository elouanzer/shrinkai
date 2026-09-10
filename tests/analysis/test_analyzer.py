import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from shrinkai.analysis.analyzer import FeatureAnalyzer, FeatureAnalyzerReport
from shrinkai.analysis.metrics import CKAMetric

# ==========================================
# MOCK CLASSES
# ==========================================


class MockExtractor(nn.Module):
    """Mocks a FeatureExtractor returning a tuple (logits, features_dict)."""

    def forward(self, x):
        # Logits are just the input, features_dict simulates two intermediate layers
        # Adding spatial dimensions (B, C, H, W) to support SpatialAttention tests
        b_size = x.size(0)
        logits = x
        features_dict = {
            "stage_1": torch.arange(b_size * 16 * 8 * 8, dtype=torch.float32).view(
                b_size, 16, 8, 8
            ),
            "stage_2": torch.arange(b_size * 32 * 4 * 4, dtype=torch.float32).view(
                b_size, 32, 4, 4
            ),
        }
        return logits, features_dict


# ==========================================
# Analyzer
# ==========================================


def test_feature_analyzer_report_show():
    """Verifies that the rich dashboard rendering does not crash."""
    report = FeatureAnalyzerReport({"stage_1": {"CKA (Linear)": 0.85, "RSA (Pearson)": 0.90}})

    # This should print to console without throwing any Exceptions
    try:
        report.show()
    except Exception as e:
        pytest.fail(f"report.show() raised an exception: {e}")


def test_feature_analyzer_evaluation():
    """Verifies that the analyzer iterates over a DataLoader and aggregates metrics."""
    # 1. Setup mock models
    teacher = MockExtractor()
    student = MockExtractor()

    # 2. Setup mock dataset (20 samples, batch_size=10 -> 2 batches)
    inputs = torch.randn(20, 5)
    targets = torch.zeros(20)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=10)

    # 3. Initialize Analyzer (force CPU for tests)
    analyzer = FeatureAnalyzer(teacher_extractor=teacher, student_extractor=student, device="cpu")

    # 4. Run evaluation with multiple metrics (string-based)
    report = analyzer.evaluate(dataloader, metrics=["cka", "attention"])

    # 5. Assertions
    scores = report.layer_scores

    # Check that both layers were intercepted
    assert "stage_1" in scores
    assert "stage_2" in scores

    # Check that both metrics were computed for stage_1
    assert "CKA (Linear)" in scores["stage_1"]
    assert "Spatial Attention" in scores["stage_1"]

    # Since both MockExtractors return identical torch.ones(...), CKA should be exactly 1.0
    assert scores["stage_1"]["CKA (Linear)"] == pytest.approx(1.0, rel=1e-4)


def test_feature_analyzer_unknown_metric():
    """Verifies that asking for an unregistered metric raises a clear ValueError."""
    analyzer = FeatureAnalyzer(MockExtractor(), MockExtractor(), device="cpu")

    # Empty dataloader is fine, it should crash before iteration
    dataset = TensorDataset(torch.randn(2, 2), torch.zeros(2))
    dataloader = DataLoader(dataset, batch_size=2)

    with pytest.raises(ValueError, match="Unknown metric 'invalid_metric'"):
        analyzer.evaluate(dataloader, metrics=["cka", "invalid_metric"])


def test_feature_analyzer_accepts_metric_instances():
    """Verifies that the analyzer accepts instantiated metric objects, not just strings."""
    analyzer = FeatureAnalyzer(MockExtractor(), MockExtractor(), device="cpu")
    dataset = TensorDataset(torch.randn(2, 2), torch.zeros(2))
    dataloader = DataLoader(dataset, batch_size=2)

    # Passing an instantiated CKAMetric instead of the string "cka"
    my_cka = CKAMetric()
    report = analyzer.evaluate(dataloader, metrics=[my_cka])

    assert "CKA (Linear)" in report.layer_scores["stage_1"]
