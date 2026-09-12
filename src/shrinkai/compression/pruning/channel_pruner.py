"""Physical (dependency-aware) structured channel pruning.

Unlike `Pruner` (which only zeroes out weights via `torch.nn.utils.prune` masks,
without changing tensor shapes or producing real latency/FLOPs gains), `ChannelPruner`
actually rebuilds smaller `Conv2d`/`Linear` (and dependent `BatchNorm`) layers with the
pruned channels physically removed.

Determining which layers can be safely shrunk together requires knowing the model's
real dataflow graph — removing a layer's output channels is only valid if every
downstream consumer of that output has its input channels shrunk to match. This module
uses `torch.fx` to trace that graph and `torch.fx.passes.shape_prop.ShapeProp` to know
each node's actual tensor shape.

Scope (by design, to stay correct rather than merely "not crashing"):
    - Supports simple, non-branching chains: a targeted `Conv2d`/`Linear` layer whose
      output feeds, without branching (exactly one consumer at every hop), through any
      number of `BatchNorm`/activation/pooling/dropout layers, into either another
      `Conv2d`/`Linear` layer or the model's final output.
    - A flatten/view/reshape between a 4D conv-style tensor and a `Linear` layer is only
      allowed once the spatial dimensions have already been reduced to 1x1 (e.g. by
      `AdaptiveAvgPool2d(1)`) — at that point each channel maps to exactly one flattened
      feature, so no interleaving ambiguity exists.
    - Grouped/depthwise convolutions (`groups != 1`) are rejected: removing channels
      from a grouped conv can change which input channels feed which output channels,
      which this implementation does not attempt to resolve.
    - Any branching topology (residual/skip connections, concatenation, attention,
      multiple consumers of the same tensor) is rejected with a clear error rather than
      silently producing an incorrect model. For those architectures, use `Pruner`
      (mask-based, safe on any topology, but without physical shrinkage) instead.
"""

import logging

import torch
import torch.nn as nn
from torch.fx.passes.shape_prop import ShapeProp
from torch.utils.data import DataLoader

from ...profiler.accuracy import compute_accuracy
from ...profiler.benchmark import BenchmarkReport, Profiler

logger = logging.getLogger(__name__)

_CHANNEL_PRESERVING_MODULE_TYPES = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.GELU,
    nn.SiLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.Hardswish,
    nn.Hardtanh,
    nn.ELU,
    nn.Mish,
    nn.Softplus,
    nn.Identity,
    nn.Dropout,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveMaxPool1d,
    nn.AdaptiveMaxPool2d,
    nn.AdaptiveMaxPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
)

_CHANNEL_PRESERVING_FUNCTION_TARGETS = {
    torch.relu,
    torch.sigmoid,
    torch.tanh,
    torch.nn.functional.relu,
    torch.nn.functional.relu6,
    torch.nn.functional.leaky_relu,
    torch.nn.functional.gelu,
    torch.nn.functional.silu,
    torch.nn.functional.sigmoid,
    torch.nn.functional.tanh,
    torch.nn.functional.hardswish,
    torch.nn.functional.hardtanh,
    torch.nn.functional.elu,
    torch.nn.functional.mish,
    torch.nn.functional.softplus,
    torch.nn.functional.dropout,
}

_RESHAPE_TARGETS = {torch.flatten, "flatten", "view", "reshape", "squeeze", "contiguous"}

_BATCHNORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_PRUNABLE_TYPES = (nn.Conv2d, nn.Linear)


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replaces the submodule at dotted path `name` with `new_module`, in place."""
    *parent_parts, leaf = name.split(".")
    parent = model
    for part in parent_parts:
        parent = getattr(parent, part)
    setattr(parent, leaf, new_module)


def _rank_keep_indices(weight: torch.Tensor, amount: float) -> torch.Tensor:
    """Ranks output units of `weight` by L2-norm and returns indices to keep."""
    out_features = weight.shape[0]
    num_prune = int(out_features * amount)

    if num_prune <= 0:
        return torch.arange(out_features)
    if num_prune >= out_features:
        raise ValueError(
            f"Cannot prune {num_prune}/{out_features} channels: at least one "
            "channel must remain. Lower `amount`."
        )

    norms = weight.detach().view(out_features, -1).norm(p=2, dim=1)
    prune_indices = torch.argsort(norms)[:num_prune]
    keep_mask = torch.ones(out_features, dtype=torch.bool)
    keep_mask[prune_indices] = False
    return torch.nonzero(keep_mask, as_tuple=False).squeeze(1)


def _shrink_output(
    layer: nn.Conv2d | nn.Linear, keep_indices: torch.Tensor
) -> nn.Conv2d | nn.Linear:
    """Rebuilds `layer` with only `keep_indices` output channels/neurons."""
    if isinstance(layer, nn.Conv2d):
        new_layer = nn.Conv2d(
            in_channels=layer.in_channels,
            out_channels=len(keep_indices),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
        )
    else:
        new_layer = nn.Linear(
            in_features=layer.in_features,
            out_features=len(keep_indices),
            bias=layer.bias is not None,
        )

    new_layer = new_layer.to(device=layer.weight.device, dtype=layer.weight.dtype)
    new_layer.train(layer.training)
    with torch.no_grad():
        new_layer.weight.copy_(layer.weight[keep_indices])
        if layer.bias is not None:
            new_layer.bias.copy_(layer.bias[keep_indices])
    return new_layer


def _shrink_input(
    layer: nn.Conv2d | nn.Linear, keep_indices: torch.Tensor
) -> nn.Conv2d | nn.Linear:
    """Rebuilds `layer` to only accept `keep_indices` input channels/features."""
    if isinstance(layer, nn.Conv2d):
        new_layer = nn.Conv2d(
            in_channels=len(keep_indices),
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=layer.groups,
            bias=layer.bias is not None,
            padding_mode=layer.padding_mode,
        )
    else:
        new_layer = nn.Linear(
            in_features=len(keep_indices),
            out_features=layer.out_features,
            bias=layer.bias is not None,
        )

    new_layer = new_layer.to(device=layer.weight.device, dtype=layer.weight.dtype)
    new_layer.train(layer.training)
    with torch.no_grad():
        new_layer.weight.copy_(layer.weight[:, keep_indices])
        if layer.bias is not None:
            new_layer.bias.copy_(layer.bias)
    return new_layer


def _shrink_batchnorm(bn: nn.Module, keep_indices: torch.Tensor) -> nn.Module:
    """Rebuilds a BatchNorm layer to only track `keep_indices` channels."""
    new_bn = type(bn)(
        num_features=len(keep_indices),
        eps=bn.eps,
        momentum=bn.momentum,
        affine=bn.affine,
        track_running_stats=bn.track_running_stats,
    )
    reference = next(bn.parameters(), None)
    if reference is None:
        reference = next(bn.buffers(), None)
    if reference is not None:
        new_bn = new_bn.to(device=reference.device, dtype=reference.dtype)
    new_bn.train(bn.training)
    with torch.no_grad():
        if bn.affine:
            new_bn.weight.copy_(bn.weight[keep_indices])
            new_bn.bias.copy_(bn.bias[keep_indices])
        if bn.track_running_stats:
            new_bn.running_mean.copy_(bn.running_mean[keep_indices])
            new_bn.running_var.copy_(bn.running_var[keep_indices])
    return new_bn


class ChannelPruner:
    """Physically removes pruned output channels from `Conv2d`/`Linear` layers.

    See the module docstring for the exact topologies this supports and rejects.

    Args:
        amount: Fraction of output channels/neurons to remove from each targeted
            layer, ranked by L2-norm (lowest-norm channels removed first). Must be
            in (0.0, 1.0). Defaults to 0.3.
    """

    def __init__(self, amount: float = 0.3) -> None:
        if not (0.0 < amount < 1.0):
            raise ValueError(f"amount must be between 0.0 and 1.0, got {amount}")
        self.amount = amount

    def apply(
        self,
        model: nn.Module,
        target_layers: list[str],
        sample_input: torch.Tensor,
    ) -> nn.Module:
        """Physically prunes the given layers, propagating the shrink downstream.

        Args:
            model: The model to prune. Mutated in place (its pruned submodules are
                replaced) and also returned for convenience.
            target_layers: Names (as in `model.named_modules()`) of `Conv2d`/`Linear`
                layers whose output channels/neurons should be pruned.
            sample_input: A representative input tensor, used to trace the model's
                dataflow graph and determine each node's actual tensor shape. Not
                used for any weight-affecting computation.

        Returns:
            nn.Module: The same `model`, with the targeted layers (and any
            dependent BatchNorm / downstream layer) physically shrunk.

        Raises:
            ValueError: If a target layer does not exist, is not a `Conv2d`/`Linear`
                with `groups == 1`, or its output cannot be safely traced to a
                single downstream layer (branching, unsupported op, or an unsafe
                flatten across non-unit spatial dimensions).
        """
        traced = torch.fx.symbolic_trace(model)
        ShapeProp(traced).propagate(sample_input)
        node_by_target = {n.target: n for n in traced.graph.nodes if n.op == "call_module"}
        # Used only to type-check nodes while walking the graph (Conv2d vs BatchNorm
        # vs activation, ...). Structurally identical to `model`'s own module tree,
        # so it stays valid for type-checking even after `model` is mutated below.
        traced_modules = dict(traced.named_modules())

        for layer_name in target_layers:
            self._prune_one_layer(model, layer_name, node_by_target, traced_modules)

        return model

    def _prune_one_layer(
        self,
        model: nn.Module,
        layer_name: str,
        node_by_target: dict,
        traced_modules: dict,
    ) -> None:
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise ValueError(f"Layer '{layer_name}' not found in the model.")
        if layer_name not in node_by_target:
            raise ValueError(
                f"Layer '{layer_name}' was not found as a traced `call_module` node. "
                "It may be unused in the forward pass, or called in a way "
                "`torch.fx.symbolic_trace` could not capture."
            )

        layer = modules[layer_name]
        if not isinstance(layer, _PRUNABLE_TYPES):
            raise ValueError(
                f"Layer '{layer_name}' is a {type(layer).__name__}, but ChannelPruner "
                "only supports Conv2d and Linear layers."
            )
        if isinstance(layer, nn.Conv2d) and layer.groups != 1:
            raise ValueError(
                f"Layer '{layer_name}' is a grouped/depthwise convolution (groups="
                f"{layer.groups}), which ChannelPruner does not support."
            )

        bn_name, downstream_name = self._walk_dependents(node_by_target[layer_name], traced_modules)

        keep_indices = _rank_keep_indices(layer.weight, self.amount)
        logger.info(
            "ChannelPruner: pruning '%s' from %d to %d output channels.",
            layer_name,
            layer.weight.shape[0],
            len(keep_indices),
        )
        _set_module_by_name(model, layer_name, _shrink_output(layer, keep_indices))

        if bn_name is not None:
            bn_layer = dict(model.named_modules())[bn_name]
            _set_module_by_name(model, bn_name, _shrink_batchnorm(bn_layer, keep_indices))

        if downstream_name is not None:
            downstream_layer = dict(model.named_modules())[downstream_name]
            if isinstance(downstream_layer, nn.Conv2d) and downstream_layer.groups != 1:
                raise ValueError(
                    f"Downstream layer '{downstream_name}' of '{layer_name}' is a "
                    f"grouped/depthwise convolution (groups={downstream_layer.groups}), "
                    "which ChannelPruner does not support."
                )
            _set_module_by_name(
                model, downstream_name, _shrink_input(downstream_layer, keep_indices)
            )

    def _walk_dependents(self, node, traced_modules: dict) -> tuple[str | None, str | None]:
        """Walks forward from `node` to find a co-prunable BatchNorm and the next
        `Conv2d`/`Linear` layer whose input channels must shrink to match.

        Args:
            node: The `call_module` FX node of the layer being pruned.
            traced_modules: `dict(traced_graph_module.named_modules())`, used to
                type-check the modules encountered while walking forward.

        Returns:
            tuple[str | None, str | None]: (batchnorm_layer_name, downstream_layer_name),
            either of which may be None (no BatchNorm in the path / target is the
            model's final layer).
        """
        current = node
        bn_name: str | None = None

        while True:
            if len(current.users) != 1:
                raise ValueError(
                    f"Layer '{node.target}' output is consumed by "
                    f"{len(current.users)} node(s) (expected exactly 1). Branching "
                    "topologies (skip connections, concatenation, ...) are not "
                    "supported by ChannelPruner; use `Pruner` instead."
                )

            next_node = next(iter(current.users))

            if next_node.op == "output":
                return bn_name, None

            if next_node.op == "call_module":
                next_module = traced_modules[next_node.target]
                if isinstance(next_module, _BATCHNORM_TYPES):
                    bn_name = next_node.target
                    current = next_node
                    continue
                if isinstance(next_module, _PRUNABLE_TYPES):
                    return bn_name, next_node.target
                if isinstance(next_module, _CHANNEL_PRESERVING_MODULE_TYPES):
                    current = next_node
                    continue
                raise ValueError(
                    f"Layer '{node.target}' output passes through unsupported "
                    f"module '{next_node.target}' ({type(next_module).__name__}) "
                    "before reaching a Conv2d/Linear layer. ChannelPruner cannot "
                    "guarantee this is safe to prune through."
                )

            if next_node.op in ("call_function", "call_method"):
                if next_node.target in _RESHAPE_TARGETS:
                    shape = current.meta["tensor_meta"].shape
                    if len(shape) > 2 and any(dim != 1 for dim in shape[2:]):
                        raise ValueError(
                            f"Layer '{node.target}' output is reshaped while its "
                            f"spatial dimensions are {tuple(shape[2:])}, not all 1. "
                            "Pruning through a flatten/view/reshape is only safe "
                            "once spatial size has been reduced to 1x1 (e.g. via "
                            "AdaptiveAvgPool2d(1))."
                        )
                    current = next_node
                    continue
                if next_node.target in _CHANNEL_PRESERVING_FUNCTION_TARGETS:
                    current = next_node
                    continue

            raise ValueError(
                f"Layer '{node.target}' output passes through unsupported "
                f"{next_node.op} '{next_node.target}' before reaching a "
                "Conv2d/Linear layer. ChannelPruner cannot guarantee this is safe "
                "to prune through; use `Pruner` instead for this topology."
            )

    def benchmark(
        self,
        original_model: nn.Module,
        pruned_model: nn.Module,
        sample_input: torch.Tensor,
        original_name: str = "Original (Dense)",
        pruned_name: str = "Pruned (Physically Shrunk)",
        val_dataloader: DataLoader | None = None,
        device: str | torch.device = "cpu",
        compute_flops: bool = False,
    ) -> BenchmarkReport:
        """Compares the original model against the physically pruned one.

        Unlike `Pruner.benchmark`, no `.finalize()`-style step is needed first:
        `pruned_model` (as returned by `.apply()`) already has fewer parameters,
        so real gains in size, latency, and FLOPs are expected here — not just
        theoretical sparsity.

        Args:
            original_model: The original, unpruned model.
            pruned_model: The model returned by `.apply()`.
            sample_input: A dummy tensor for latency measurement.
            original_name: Display label for the original model.
            pruned_name: Display label for the pruned model.
            val_dataloader: Optional dataloader for accuracy comparison.
            device: Device for the benchmark.
            compute_flops: If True, also reports FLOPs per sample for both
                models. Safe to enable here (unlike for `Pruner`/`Quantizer`),
                since physical channel pruning keeps standard Conv2d/Linear ops
                that FLOPs counting handles accurately. Defaults to False.

        Returns:
            BenchmarkReport: Structured benchmark report ready for `.show()`.
        """
        original_acc: float | None = None
        pruned_acc: float | None = None

        if val_dataloader is not None:
            original_acc = compute_accuracy(original_model, val_dataloader, device)
            pruned_acc = compute_accuracy(pruned_model, val_dataloader, device)

        return Profiler.compare(
            teacher=original_model,
            student=pruned_model,
            sample_input=sample_input,
            device=device,
            teacher_name=original_name,
            student_name=pruned_name,
            teacher_acc=original_acc,
            student_acc=pruned_acc,
            compute_flops=compute_flops,
        )
