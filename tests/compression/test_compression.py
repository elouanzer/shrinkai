import warnings

import pytest
import torch
import torch.nn as nn

from shrinkai.compression.quantization import QuantConfig, Quantizer

warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.ao")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.ao")

# ==========================================
# FIXTURES
# ==========================================


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def sample_input():
    return torch.randn(1, 10)


@pytest.fixture(autouse=True)
def setup_quantization_engine():
    supported_engines = torch.backends.quantized.supported_engines
    if "fbgemm" in supported_engines:
        torch.backends.quantized.engine = "fbgemm"
    elif "qnnpack" in supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    else:
        pytest.skip("No engine for this device.")


# ==========================================
# Quantization
# ==========================================


def test_quantconfig_validation():
    """Test that invalid configurations raise appropriate errors."""
    with pytest.raises(ValueError, match="Invalid strategy"):
        QuantConfig(strategy="invalid_strategy")


def test_ptq_application(model, sample_input):
    """Test Post-Training Quantization dynamically converts Linear layers."""
    config = QuantConfig(target_dtype="int8", strategy="ptq")
    quantizer = Quantizer(config)

    quantized_model = quantizer.apply(model)
    output = quantized_model(sample_input)

    # Assert forward pass works
    assert output.shape == (1, 5)

    # In PyTorch dynamic PTQ, nn.Linear is swapped for DynamicQuantizedLinear
    is_quantized = any(
        isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
        for module in quantized_model.modules()
    )
    assert is_quantized, "Linear layers were not dynamically quantized."


def test_qat_application_and_finalize(model, sample_input):
    """Test QAT inserts FakeQuantize nodes and finalize bakes them."""
    config = QuantConfig(target_dtype="int8", strategy="qat")
    quantizer = Quantizer(config)

    qat_model = quantizer.apply(model)

    # Verify QAT preparation
    has_qat_layers = any(
        isinstance(module, torch.ao.nn.qat.Linear) for module in qat_model.modules()
    )
    assert has_qat_layers, "QAT layers were not inserted."

    # Verify forward pass during QAT (simulated quantization)
    qat_output = qat_model(sample_input)
    assert qat_output.shape == (1, 5)

    # Finalize the model
    final_model = Quantizer.finalize_qat(qat_model)

    # Verify finalizing converts QAT layers to fully quantized layers
    is_finalized = any(
        isinstance(module, torch.ao.nn.quantized.Linear) for module in final_model.modules()
    )
    assert is_finalized, "FakeQuantize nodes were not converted to real quantized weights."
