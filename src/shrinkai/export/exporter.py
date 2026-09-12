"""Deployment export utilities: get a trained/compressed model out of the PyTorch
Python process and into a format runnable on other runtimes or edge devices.

Works on any `nn.Module`, or one already pruned/quantized via
`shrinkai.compression`, or a `Distiller`'s trained student.
"""

from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn


def export_onnx(
    model: nn.Module,
    sample_input: torch.Tensor | tuple[torch.Tensor, ...],
    path: str | Path,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_batch: bool = True,
    opset_version: int = 17,
) -> Path:
    """Exports a model to ONNX, for inference on ONNX Runtime or other non-PyTorch
    engines (many mobile/edge inference stacks consume ONNX).

    Note:
        Uses PyTorch's legacy TorchScript-based exporter (`dynamo=False`) rather than
        the newer `torch.export`-based one, since the latter additionally requires the
        `onnxscript` package. As of PyTorch 2.9+, this legacy exporter is itself
        deprecated upstream (a `DeprecationWarning` is expected) in favor of the
        `torch.export`, it still works correctly today, but migrating this
        function once `onnxscript` is a lighter/more stable dependency is a known
        next step.

    Requires the optional `onnx` package: `pip install shrinkai[export]`.

    Args:
        model: The model to export. Switched to eval mode internally.
        sample_input: A representative input tensor (or tuple of tensors, for
            multi-input models) used to trace the model's operations.
        path: Destination `.onnx` file path.
        input_names: Optional names for the graph's input nodes. If omitted while
            `dynamic_batch=True`, names are auto-generated (`input_0`, `input_1`, ...)
            so the dynamic batch axis can be declared.
        output_names: Optional names for the graph's output nodes.
        dynamic_batch: If True (default), dimension 0 of every input is marked
            dynamic, so the exported graph accepts any batch size at inference
            time (PyTorch's shape inference propagates this to the outputs too).
            If False, the graph is frozen to `sample_input`'s exact batch size.
        opset_version: Target ONNX opset version. Defaults to 17.

    Returns:
        Path: The path the model was exported to.

    Raises:
        ImportError: If the optional `onnx` package is not installed.
    """
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ONNX export requires the optional 'onnx' package. Install it with "
            "`pip install shrinkai[export]` (or `pip install onnx`)."
        ) from exc

    model.eval()
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    args = sample_input if isinstance(sample_input, tuple) else (sample_input,)

    dynamic_axes = None
    if dynamic_batch:
        if input_names is None:
            input_names = [f"input_{i}" for i in range(len(args))]
        dynamic_axes = {name: {0: "batch_size"} for name in input_names}

    torch.onnx.export(
        model,
        args,
        str(save_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        dynamo=False,
    )
    return save_path


def export_torchscript(
    model: nn.Module,
    path: str | Path,
    sample_input: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
    method: Literal["trace", "script"] = "trace",
) -> Path:
    """Exports a model to TorchScript, for deployment outside a Python process
    (the LibTorch C++ runtime, PyTorch Mobile, ...).

    Note:
        `torch.jit.trace`/`torch.jit.script` are flagged as deprecated upstream
        in favor of `torch.export` (a `DeprecationWarning` is expected). They are
        used here anyway because TorchScript `.pt` files remain, as of this
        writing, the format most consistently supported by LibTorch C++ and
        PyTorch Mobile in practice; migrating to `torch.export` once its own
        deployment story (C++/mobile loading) is verified as a solid replacement
        is being followed-up.

    Args:
        model: The model to export. Switched to eval mode internally.
        path: Destination file path (conventionally `.pt`).
        sample_input: Required when `method="trace"`: a representative input
            tensor (or tuple of tensors) used to record the forward pass. Ignored
            when `method="script"`.
        method: "trace" (default) records the actual tensor operations executed
            for `sample_input`, fast and broadly compatible, but silently
            "bakes in" any data-dependent control flow (e.g. a Python `if` on a
            tensor's value) as if it always took the same branch. "script"
            compiles the model's Python source through TorchScript's
            subset-of-Python compiler, correctly preserving control flow, but
            requires the model's code to be scriptable.

    Returns:
        Path: The path the model was exported to.

    Raises:
        ValueError: If `method` is not "trace"/"script", or if `method="trace"`
            and `sample_input` is None.
    """
    if method not in ("trace", "script"):
        raise ValueError(f"method must be 'trace' or 'script', got '{method}'.")

    model.eval()
    save_path = Path(path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if method == "trace":
        if sample_input is None:
            raise ValueError("sample_input is required when method='trace'.")
        args = sample_input if isinstance(sample_input, tuple) else (sample_input,)
        with torch.no_grad():
            exported = torch.jit.trace(model, args)
    else:
        exported = torch.jit.script(model)

    exported.save(str(save_path))
    return save_path
