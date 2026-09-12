import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from shrinkai.export import export_onnx, export_torchscript

# ==========================================
# 1. MOCK MODEL
# ==========================================


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 4)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model_and_sample():
    return TinyModel().eval(), torch.randn(2, 10)


# ==========================================
# 2. ONNX EXPORT
# ==========================================


def test_export_onnx_creates_valid_file(model_and_sample, tmp_path: Path):
    """Verifies the exported file exists and passes ONNX's own structural checker."""
    import onnx

    model, sample = model_and_sample
    out_path = tmp_path / "model.onnx"

    result_path = export_onnx(model, sample, out_path)

    assert result_path == out_path
    assert out_path.exists()
    onnx.checker.check_model(onnx.load(str(out_path)))


def test_export_onnx_dynamic_batch_propagates_to_output(model_and_sample, tmp_path: Path):
    """Verifies dynamic_batch=True marks the batch dim as dynamic on inputs, and
    that this propagates to the output shape via ONNX shape inference.
    """
    import onnx
    from onnx import shape_inference

    model, sample = model_and_sample
    out_path = tmp_path / "dynamic.onnx"

    export_onnx(model, sample, out_path, input_names=["input"], dynamic_batch=True)

    onnx_model = onnx.load(str(out_path))
    inferred = shape_inference.infer_shapes(onnx_model)
    output_dim0 = inferred.graph.output[0].type.tensor_type.shape.dim[0]
    assert output_dim0.dim_param  # symbolic, not a fixed dim_value


def test_export_onnx_static_batch_freezes_shape(model_and_sample, tmp_path: Path):
    """Verifies dynamic_batch=False freezes the graph to sample_input's batch size."""
    import onnx

    model, sample = model_and_sample
    out_path = tmp_path / "static.onnx"

    export_onnx(model, sample, out_path, input_names=["input"], dynamic_batch=False)

    onnx_model = onnx.load(str(out_path))
    input_dim0 = onnx_model.graph.input[0].type.tensor_type.shape.dim[0]
    assert input_dim0.dim_value == 2  # sample batch size, not symbolic


def test_export_onnx_raises_clear_error_without_onnx_package(
    model_and_sample, tmp_path: Path, monkeypatch
):
    """Verifies a missing 'onnx' package raises a clear, actionable ImportError."""
    model, sample = model_and_sample
    monkeypatch.setitem(sys.modules, "onnx", None)

    with pytest.raises(ImportError, match="shrinkai\\[export\\]"):
        export_onnx(model, sample, tmp_path / "model.onnx")


def test_export_onnx_creates_parent_directories(model_and_sample, tmp_path: Path):
    model, sample = model_and_sample
    out_path = tmp_path / "nested" / "dir" / "model.onnx"

    export_onnx(model, sample, out_path)

    assert out_path.exists()


# ==========================================
# 3. TORCHSCRIPT EXPORT
# ==========================================


def test_export_torchscript_trace_roundtrip(model_and_sample, tmp_path: Path):
    """Verifies a traced model round-trips through save/load with matching output."""
    model, sample = model_and_sample
    out_path = tmp_path / "model_traced.pt"

    export_torchscript(model, out_path, sample_input=sample, method="trace")

    loaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        expected = model(sample)
        actual = loaded(sample)
    assert torch.allclose(expected, actual)


def test_export_torchscript_script_roundtrip(model_and_sample, tmp_path: Path):
    """Verifies a scripted model round-trips through save/load with matching output."""
    model, sample = model_and_sample
    out_path = tmp_path / "model_scripted.pt"

    export_torchscript(model, out_path, method="script")

    loaded = torch.jit.load(str(out_path))
    with torch.no_grad():
        expected = model(sample)
        actual = loaded(sample)
    assert torch.allclose(expected, actual)


def test_export_torchscript_trace_requires_sample_input(model_and_sample, tmp_path: Path):
    model, _ = model_and_sample
    with pytest.raises(ValueError, match="sample_input is required"):
        export_torchscript(model, tmp_path / "model.pt", method="trace")


def test_export_torchscript_rejects_invalid_method(model_and_sample, tmp_path: Path):
    model, sample = model_and_sample
    with pytest.raises(ValueError, match="must be 'trace' or 'script'"):
        export_torchscript(model, tmp_path / "model.pt", sample_input=sample, method="invalid")
