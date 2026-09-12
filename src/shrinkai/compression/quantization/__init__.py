"""Quantization: reducing the numerical precision of a model's weights/activations.

`Quantizer` (configured via `QuantConfig`) supports Post-Training Quantization
(PTQ, dynamic, or static when `QuantConfig.calibrate_data` is provided) and
Quantization-Aware Training (QAT, meant to be trained further, e.g. through a
`shrinkai.distillation.Distiller`, before calling `Quantizer.finalize_qat`).

Note:
    Built on `torch.ao.quantization`, which PyTorch has marked deprecated in
    favor of `torchao`. It still works correctly today; migrating this module
    once `torchao`'s API has stabilized is a known follow-up.
"""

from .quantization import QuantConfig, Quantizer

__all__ = ["QuantConfig", "Quantizer"]
