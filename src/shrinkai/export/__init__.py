"""Deployment export: get a model out of the PyTorch Python process.

`export_onnx` and `export_torchscript` work on any `nn.Module`,
one already pruned/quantized via `shrinkai.compression`, or a distilled
trained student (also reachable directly as `Distiller.export_onnx`/
`Distiller.export_torchscript`).
"""

from .exporter import export_onnx, export_torchscript

__all__ = ["export_onnx", "export_torchscript"]
