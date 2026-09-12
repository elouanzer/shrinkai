import io
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader

from ...profiler.accuracy import compute_accuracy
from ...profiler.benchmark import BenchmarkReport, Profiler

logger = logging.getLogger(__name__)


def _clone_module(model: nn.Module) -> nn.Module:
    """Deep-clones a module via a serialization round-trip.

    `copy.deepcopy` raises on modules with active `torch.nn.utils.prune`
    reparametrizations (masked weights are non-leaf tensors, which
    `Tensor.__deepcopy__` rejects). A `torch.save`/`torch.load` round-trip
    serializes the same tensors without going through that code path.
    """
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


@dataclass
class PruningConfig:
    """
    Configuration parameters for model pruning.

    Attributes:
        method (str): The pruning strategy to apply.
            - "unstructured": Removes individual weights based on L1-norm (closest to zero).
              Creates sparse tensors but doesn't change tensor shapes.
            - "structured": Zeroes entire channels/neurons based on their Ln-norm,
              via `torch.nn.utils.prune`. This does NOT change tensor shapes either:
              the pruned channels stay in memory as zeros, so it does not by itself
              reduce parameter count, model size, or inference latency on standard
              hardware. To physically remove channels (actually shrinking the
              model), use `ChannelPruner` instead.
            Defaults to "unstructured".

        amount (float): The fraction of weights/channels to prune.
            Must be a float between 0.0 and 1.0 (e.g., 0.3 means 30% pruned).
            Defaults to 0.3.

        target_types (tuple): The PyTorch module types to apply pruning to.
            Defaults to (nn.Linear, nn.Conv2d).
    """

    method: str = "unstructured"
    amount: float = 0.3
    target_types: tuple = (nn.Linear, nn.Conv2d)

    def __post_init__(self):
        valid_methods = ["unstructured", "structured"]
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method '{self.method}'. Supported choices: {valid_methods}")
        if not (0.0 < self.amount < 1.0):
            raise ValueError(f"Pruning amount must be between 0.0 and 1.0, got {self.amount}")


class Pruner:
    """
    Orchestrates the pruning of PyTorch models to induce sparsity.

    The Pruner traverses the model and applies masking to the weights according
    to the PruningConfig. It seamlessly supports "Pruning-Aware Training" since
    PyTorch pruning attaches forward pre-hooks to dynamically mask weights during
    the forward pass, allowing gradient updates on the remaining unpruned weights.

    Args:
        config (PruningConfig): The configuration object defining the pruning rules.
    """

    def __init__(self, config: PruningConfig):
        self.config = config

    def apply(self, model: nn.Module) -> nn.Module:
        """
        Applies the selected pruning strategy to the model's target layers.

        Args:
            model (nn.Module): The standard PyTorch model.

        Returns:
            nn.Module: The pruned model (with pruning hooks attached).
        """
        logger.info(f"Applying {self.config.method} pruning ({self.config.amount * 100}%)...")

        for _, module in model.named_modules():
            if isinstance(module, self.config.target_types):
                if self.config.method == "unstructured":
                    self._apply_unstructured(module)
                elif self.config.method == "structured":
                    self._apply_structured(module)

        return model

    def _apply_unstructured(self, module: nn.Module):
        """Internal method: Applies L1 Unstructured Pruning to a specific module."""
        prune.l1_unstructured(module, name="weight", amount=self.config.amount)
        if getattr(module, "bias", None) is not None:
            prune.l1_unstructured(module, name="bias", amount=self.config.amount)

    def _apply_structured(self, module: nn.Module):
        """Internal method: Applies L2 Structured Pruning (e.g., removing output channels)."""
        prune.ln_structured(module, name="weight", amount=self.config.amount, n=2, dim=0)

    @staticmethod
    def finalize(pruned_model: nn.Module) -> nn.Module:
        """
        Makes the pruning permanent by removing the PyTorch forward hooks and
        baking the sparsity mask directly into the weight tensors.

        This MUST be called after training/distillation before saving the model
        for deployment, otherwise the masking overhead slows down inference.

        Args:
            pruned_model (nn.Module): The model with pruning hooks attached.

        Returns:
            nn.Module: The permanent, clean sparse model.
        """
        if getattr(pruned_model, "_pruning_finalized", False):
            logger.warning("Pruning has already been finalized on this model. Skipping.")
            return pruned_model

        logger.info("Finalizing pruning: removing hooks and baking masks into weights...")
        hooks_removed = False
        for _, module in pruned_model.named_modules():
            if hasattr(module, "weight_mask"):
                prune.remove(module, "weight")
                hooks_removed = True
            if hasattr(module, "bias_mask"):
                prune.remove(module, "bias")
                hooks_removed = True

        if not hooks_removed:
            logger.warning("No pruning hooks were found. Was the model actually pruned?")
        pruned_model._pruning_finalized = True

        return pruned_model

    def benchmark(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module,
        sample_input: torch.Tensor,
        original_name: str = "Original (Dense)",
        pruned_name: str = "Pruned (Sparse)",
        val_dataloader: DataLoader | None = None,
        device: str | torch.device = "cpu",
    ) -> BenchmarkReport:
        """
        Evaluates and compares the performance footprint of the dense vs. pruned model.
        Note: True latency gains for unstructured pruning require sparse-tensor
        supported hardware engines.

        Args:
            original_model (nn.Module): The dense baseline model.
            pruned_model (nn.Module): The pruned model.
            sample_input (torch.Tensor): A dummy tensor for latency measurement.
            original_name (str, optional): Display label for baseline.
            pruned_name (str, optional): Display label for pruned model.
            val_dataloader (DataLoader | None, optional): Dataloader for accuracy.
            device (str | torch.device, optional): Device for the benchmark.

        Returns:
                    BenchmarkReport: Structured benchmark report ready for `.show()`.
        """
        eval_pruned_model = _clone_module(pruned_model)
        eval_pruned_model = self.finalize(eval_pruned_model)

        original_acc: float | None = None
        pruned_acc: float | None = None

        if val_dataloader is not None:
            original_acc = compute_accuracy(original_model, val_dataloader, device)
            pruned_acc = compute_accuracy(eval_pruned_model, val_dataloader, device)

        return Profiler.compare(
            teacher=original_model,
            student=eval_pruned_model,
            sample_input=sample_input,
            device=device,
            teacher_name=original_name,
            student_name=pruned_name,
            teacher_acc=original_acc,
            student_acc=pruned_acc,
        )
