"""
The `quantization`module provides tools to compress neural networks by reducing the precision
of their weights and activations, significantly decreasing memory footprint and
inference latency. It is designed to work as a standalone utility or to be
seamlessly composed with the `shrinkai.distillation` module.

Core Techniques:
    1. Post-Training Quantization (PTQ):
       A fast, training-free approach that converts a pre-trained FP32 model
       into a lower precision format (e.g., INT8) on the fly. Best for rapid
       deployment where a slight accuracy drop is acceptable.

    2. Quantization-Aware Training (QAT):
       Inserts 'Fake Quantization' nodes into the model prior to training.
       The forward pass simulates lower precision, while the backward pass
       updates weights in full precision. The model learns to mitigate quantization
       noise, resulting in higher accuracy. It could be use in Knowledge Distillation.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...profiler.accuracy import compute_accuracy
from ...profiler.benchmark import BenchmarkReport, Profiler

logger = logging.getLogger(__name__)


@dataclass
class QuantConfig:
    """
    Configuration parameters for model quantization.

    This dataclass standardizes how quantization is applied across different
    backends and strategies. It defines the target bit-width and the mathematical
    approach used to compress the network.

    Attributes:
        target_dtype (str): The target data type for the model's weights.
            Currently supported options include:
            - "int8": 8-bit integer (Standard for CPU/Edge deployment).
            - "fp16": 16-bit float (Standard for GPU memory reduction).
            Defaults to "int8".

        strategy (str): The quantization strategy to apply.
            - "ptq" (Post-Training Quantization): Applies immediate mathematical
              conversion to the weights. No gradient computation or training is required.
            - "qat" (Quantization-Aware Training): Prepares the model with FakeQuantize
              nodes. Requires subsequent training (e.g., via `shrinkai.Distiller`)
              before final conversion.
            Defaults to "ptq".

        backend (str): The underlying engine executing the quantization.
            - "torch": Native PyTorch quantization (fbgemm/qnnpack). Ideal for CNNs and small LMs.
            - "bitsandbytes": (Reserved for future LLM integration) Block-wise quantization.
            Defaults to "torch".

        calibrate_data (Any | None): A dataloader (or any iterable of batches, each
            either a plain input tensor or a `(inputs, labels, ...)` tuple/list) used
            exclusively for Static PTQ. If provided, the quantizer runs a forward
            pass over this data to calibrate activation scales before conversion.
            If None, Dynamic PTQ is used instead (weights only, no calibration).
            Defaults to None.
    """

    target_dtype: str = "int8"
    strategy: str = "ptq"
    backend: str = "torch"
    calibrate_data: Any | None = None

    def __post_init__(self):
        valid_strategies = ["ptq", "qat"]
        valid_dtype = ["int8", "fp16"]

        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{self.strategy}'. Supported choices: {valid_strategies}"
            )

        if self.target_dtype not in valid_dtype:
            raise ValueError(
                f"Invalid strategy '{self.target_dtype}'. Supported choices: {valid_dtype}"
            )


class Quantizer:
    """
    Orchestrates the quantization of PyTorch models.

    The Quantizer reads a `QuantConfig` and safely modifies the computational graph
    of a given PyTorch `nn.Module`. It handles the complexities of PyTorch's native
    quantization APIs.

    Args:
        config (QuantConfig): The configuration object defining the quantization rules.
    """

    def __init__(self, config: QuantConfig):
        self.config = config

    def apply(self, model: nn.Module) -> nn.Module:
        """
        Applies the selected quantization strategy to the model.

        Depending on `config.strategy`, this method will either immediately convert
        the weights (PTQ) or insert FakeQuantize nodes for future training (QAT).

        Args:
            model (nn.Module): The standard, full-precision PyTorch model.

        Returns:
            nn.Module: The modified model. If PTQ, it is ready for deployment.
            If QAT, it must be trained and then passed to `finalize_qat()`.
        """
        if self.config.strategy == "ptq":
            return self._apply_ptq(model)
        elif self.config.strategy == "qat":
            return self._apply_qat(model)
        else:
            raise NotImplementedError(f"Strategy {self.config.strategy} is not implemented yet.")

    def _apply_ptq(self, model: nn.Module) -> nn.Module:
        """
        Applies Post-Training Quantization (PTQ).

        Uses Static PTQ (calibrated on `config.calibrate_data`) if provided,
        otherwise falls back to Dynamic PTQ (weights only, no calibration needed).
        Ideal for immediate inference optimization without retraining.
        """
        logger.info(
            f"Applying PTQ ({self.config.target_dtype}) using {self.config.backend} backend..."
        )

        if self.config.backend != "torch" or self.config.target_dtype != "int8":
            raise NotImplementedError(
                f"PTQ is not yet supported for backend '{self.config.backend}' "
                f"with dtype '{self.config.target_dtype}'."
            )

        if self.config.calibrate_data is not None:
            return self._apply_static_ptq(model)

        return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    def _apply_static_ptq(self, model: nn.Module) -> nn.Module:
        """
        Applies Static Post-Training Quantization, calibrated on `config.calibrate_data`.

        Wraps the model with `QuantStub`/`DeQuantStub` (via `QuantWrapper`) so it keeps
        accepting and returning standard float tensors, runs a calibration pass over
        `config.calibrate_data` to observe activation ranges, then converts both
        weights and activations to int8.

        Note:
            For best accuracy, fuse Conv-BN-ReLU sequences on `model` (via
            `torch.ao.quantization.fuse_modules`) before calling `apply()`. This
            implementation works without fusion, at the cost of some accuracy.
        """
        logger.info("Calibrating activations for Static PTQ...")

        wrapped_model = torch.ao.quantization.QuantWrapper(model)
        wrapped_model.eval()

        engine = torch.backends.quantized.engine
        wrapped_model.qconfig = torch.ao.quantization.get_default_qconfig(engine)
        prepared_model = torch.ao.quantization.prepare(wrapped_model, inplace=False)

        with torch.no_grad():
            for batch in self.config.calibrate_data:
                inputs = batch[0] if isinstance(batch, list | tuple) else batch
                prepared_model(inputs)

        return torch.ao.quantization.convert(prepared_model, inplace=False)

    def _apply_qat(self, model: nn.Module) -> nn.Module:
        """
        Prepares the model for Quantization-Aware Training (QAT).
        Inserts 'Fake Quantization' nodes without converting the actual tensors,
        allowing gradients to flow during the distillation process.
        """
        logger.info(f"Preparing model for QAT ({self.config.target_dtype})...")

        if self.config.backend == "torch":
            model.train()
            engine = torch.backends.quantized.engine
            model.qconfig = torch.ao.quantization.get_default_qat_qconfig(engine)
            qat_model = torch.ao.quantization.prepare_qat(model, inplace=False)
            return qat_model
        else:
            raise NotImplementedError(
                f"QAT is not yet supported for backend '{self.config.backend}'."
            )

    @staticmethod
    def finalize_qat(qat_model: nn.Module) -> nn.Module:
        """
        To be called AFTER the distillation training loop (distiller.fit).
        Converts the simulated FakeQuantize nodes into actual quantized weights (e.g., int8).

        Args:
            qat_model (nn.Module): The model trained with FakeQuantize nodes.

        Returns:
            nn.Module: The fully quantized model.
        """
        logger.info(
            "Finalizing QAT model: converting FakeQuantize nodes to actual quantized weights..."
        )
        qat_model.eval()
        return torch.ao.quantization.convert(qat_model, inplace=False)

    def benchmark(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        sample_input: torch.Tensor,
        original_name: str = "Original (FP32)",
        quantized_name: str = "Quantized",
        val_dataloader: DataLoader | None = None,
        device: str | torch.device = "cpu",
    ) -> BenchmarkReport:
        """Runs complete profiling suite on original and quantized models.

        Args:
            original_model: The base FP32 PyTorch model.
            quantized_model: The model returned by `.apply()`.
            sample_input: Batch tensor matching target inference dimension.
            original_name: Display label for the original model.
            quantized_name: Display label for the quantized model.
            val_dataloader: Optional dataloader to compute final accuracy metrics.
            device: Device to run the benchmark on (default "cpu", as INT8 is often CPU-optimized).

        Returns:
            BenchmarkReport: Structured benchmark report ready for `.show()`.
        """
        original_acc: float | None = None
        quantized_acc: float | None = None

        eval_quantized_model = quantized_model
        if self.config.strategy == "qat":
            eval_quantized_model = self.finalize_qat(quantized_model)

        if val_dataloader is not None:
            original_acc = compute_accuracy(original_model, val_dataloader, device)
            quantized_acc = compute_accuracy(eval_quantized_model, val_dataloader, device)

        return Profiler.compare(
            teacher=original_model,
            student=eval_quantized_model,
            sample_input=sample_input,
            device=device,
            teacher_name=original_name,
            student_name=quantized_name,
            teacher_acc=original_acc,
            student_acc=quantized_acc,
        )
