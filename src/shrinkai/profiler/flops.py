"""FLOPs (floating point operations) counting for edge-deployment profiling.

Unlike parameter count or disk size, FLOPs are hardware-independent: they
characterize the amount of compute a forward pass requires.
"""

import logging

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from ..utils import resolve_device

logger = logging.getLogger(__name__)


def count_flops(
    model: nn.Module,
    sample_input: torch.Tensor | tuple[torch.Tensor, ...],
    device: torch.device | str = "auto",
) -> int:
    """Counts the total FLOPs of one forward pass of `model` on `sample_input`.

    Uses `torch.utils.flop_counter.FlopCounterMode`, which instruments the actual
    tensor operations dispatched during the forward pass, covering standard
    layers (Conv, Linear, matmul, attention, ...) precisely, rather than a
    hand-maintained per-layer-type formula. 1 multiply-add (MAC) is counted as 2
    FLOPs, matching the usual convention.

    Note:
        Custom/opaque kernels (e.g. the quantized ops produced by
        `shrinkai.compression.quantization.Quantizer`) have no FLOPs formula
        registered and are silently counted as 0. A warning is logged if the
        total comes back as 0 despite the model having parameters, since that
        usually signals an undercount rather than a genuinely free model.

    Args:
        model: Model to profile.
        sample_input: Representative input tensor (or tuple of tensors, for
            multi-input models) matching the model's forward signature.
        device: Device to run the single, untimed forward pass on. FLOPs are a
            static property of the computation graph and do not depend on the
            device; this only needs to be a device the model can actually run on.

    Returns:
        int: Total FLOPs for one forward pass on `sample_input`.
    """
    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    args = sample_input if isinstance(sample_input, tuple) else (sample_input,)
    args = tuple(a.to(resolved_device) if isinstance(a, torch.Tensor) else a for a in args)

    was_training = model.training
    model.eval()

    with FlopCounterMode(display=False) as flop_counter:
        with torch.no_grad():
            model(*args)

    model.train(was_training)

    total_flops = flop_counter.get_total_flops()
    if total_flops == 0 and any(p.numel() > 0 for p in model.parameters()):
        logger.warning(
            "count_flops returned 0 for a model with parameters. This usually "
            "means it uses custom/opaque ops (e.g. quantized kernels) that "
            "FlopCounterMode cannot see through, not that it is actually free."
        )

    return total_flops
