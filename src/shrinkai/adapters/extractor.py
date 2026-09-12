from collections.abc import Iterable, Mapping
from typing import Any

import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    """Wraps a PyTorch model to extract intermediate feature maps via forward hooks.

    This wrapper does not modify the original model's source code. It dynamically
    attaches hooks to intercept the output of specified layers during the forward pass.
    """

    def __init__(self, model: nn.Module, target_layers: Iterable[str] | Mapping[str, str]) -> None:
        """Initializes the FeatureExtractor.

        Args:
            model: The PyTorch model to extract features from.
            target_layers: An iterable of layer names to hook (e.g., ["layer1", "layer2"]),
                or a mapping dictionary to assign common aliases for cross-model matching
                (e.g., {"layer4.conv1": "block_1", "layer8.conv1": "block_2"}).

        Raises:
            ValueError: If a target layer does not exist in the model.
        """
        super().__init__()
        self.model = model
        if isinstance(target_layers, Mapping):
            aliases = list(target_layers.values())
            if len(set(aliases)) != len(aliases):
                raise ValueError("Many layers have the same alias.")
            self.target_layers = target_layers
        else:
            self.target_layers = {layer: layer for layer in target_layers}
        self.features: dict[str, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        """Registers PyTorch forward hooks on the target layers."""
        model_modules = dict(self.model.named_modules())

        for layer_name, alias in self.target_layers.items():
            if layer_name not in model_modules:
                raise ValueError(
                    f"Layer '{layer_name}' not found in the model. "
                    f"Available layers: {list(model_modules.keys())[:10]}..."
                )

            module = model_modules[layer_name]

            def hook_fn(
                module: nn.Module, input: tuple, output: torch.Tensor, name: str = alias
            ) -> None:
                if isinstance(output, tuple):
                    output = output[0]  # take main tensor in first position
                self.features[name] = output

            handle = module.register_forward_hook(hook_fn)
            self._hooks.append(handle)

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Performs a forward pass and captures intermediate features.

        Args:
            x: Input tensor.
            *args: Additional positional arguments for the model.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            tuple[torch.Tensor, dict[str, torch.Tensor]]: A tuple containing the
                final model output (logits) and a dictionary of extracted features.
        """
        self.features.clear()
        logits = self.model(x, *args, **kwargs)
        return logits, self.features.copy()

    def remove_hooks(self) -> None:
        """Removes all registered hooks to prevent memory leaks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()
