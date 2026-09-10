import warnings
from typing import Any

import torch
import torch.nn as nn


class FeatureProjector(nn.Module):
    """Projects student features to match the channel dimensions of the teacher features.

    Uses 1x1 convolutions for spatial feature maps (B, C, H, W) or Linear layers
    for flattened features (B, C) or sequences (B, L, D).
    """

    def __init__(self, mapping_config: dict[str, dict[str, Any]]) -> None:
        """Initializes the FeatureProjector.

        Args:
            mapping_config: A dictionary defining the projection parameters for each layer.
                Format: {
                    "block_1": {"in_channels": 64, "out_channels": 128, "type": "conv", "use_norm": True},
                    "block_2": {"in_channels": 256, "out_channels": 512, "type": "linear"}
                }
                - `type`: Must be either 'conv' (default) or 'linear'.
                - `use_norm`: (Optional) If True, appends a normalization layer (BatchNorm2d
                for 'conv', LayerNorm for 'linear') to stabilize gradients. Defaults to False.
        """  # noqa: E501
        super().__init__()
        self.projectors = nn.ModuleDict()

        for layer_name, config in mapping_config.items():
            if "in_channels" not in config:
                raise ValueError(
                    f"Missing required key 'in_channels' for layer '{layer_name}'in mapping_config."
                )
            if "out_channels" not in config:
                raise ValueError(
                    f"Missing required key 'out_channels' for layer '{layer_name}'"
                    "in mapping_config."
                )

            in_c = config["in_channels"]
            out_c = config["out_channels"]
            proj_type = config.get("type", "conv")
            use_norm = config.get("use_norm", False)

            layers = []
            if proj_type == "conv":
                layers.append(nn.Conv2d(in_c, out_c, kernel_size=1, bias=not use_norm))
                if use_norm:
                    layers.append(nn.BatchNorm2d(out_c))
            elif proj_type == "linear":
                layers.append(nn.Linear(in_c, out_c, bias=not use_norm))
                if use_norm:
                    layers.append(nn.LayerNorm(out_c))
            else:
                raise ValueError(f"Unsupported projection type '{proj_type}'.")

            safe_key = layer_name.replace(".", "_")
            self.projectors[safe_key] = nn.Sequential(*layers)

        self.mapping_keys = {k: k.replace(".", "_") for k in mapping_config.keys()}

    def forward(self, student_features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Applies the projection layers to the student's extracted features.

        Args:
            student_features: Dictionary of raw feature tensors from the student.

        Returns:
            dict[str, torch.Tensor]: Dictionary of projected feature tensors.
        """
        projected_features = {}

        for layer_name, feature_tensor in student_features.items():
            safe_key = self.mapping_keys.get(layer_name)

            if safe_key and safe_key in self.projectors:
                projected_features[layer_name] = self.projectors[safe_key](feature_tensor)
            else:
                warnings.warn(
                    f"No projection configured for layer '{layer_name}'. "
                    "Feature tensor is returned. Check 'mapping_config' if this is unexpected.",
                    stacklevel=2,
                )
                projected_features[layer_name] = feature_tensor

        return projected_features


class AttentionHeadSelector(nn.Module):
    """Adapts teacher attention maps to match student dimensions by selecting specific heads.

    In Transformer distillation (e.g., TinyBERT), a student often has fewer attention
    heads than the teacher (e.g., 2 vs 12). This adapter slices the teacher's attention
    tensors [Batch, Heads, Seq, Seq] to keep only the indices corresponding to the student.
    """

    def __init__(self, heads_to_keep: list[int]) -> None:
        """Initializes the AttentionHeadSelector.

        Args:
            heads_to_keep: List of integer indices representing which teacher heads
                to retain (e.g., [0, 6] to keep the first and seventh head).
        """
        super().__init__()
        self.heads_to_keep = heads_to_keep

    def forward(self, attention_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Slices the attention tensors along the head dimension.

        Args:
            attention_dict: Dictionary of attention tensors of shape [B, Num_Heads, S, S].

        Returns:
            dict[str, torch.Tensor]: Dictionary of sliced tensors of
                shape [B, len(heads_to_keep), S, S].
        """
        projected_attention = {}
        for layer_name, tensor in attention_dict.items():
            if tensor.dim() != 4:
                raise ValueError(
                    f"Expected 4D attention tensor [Batch, Heads, Seq, Seq], "
                    f"but got shape {tensor.shape} for layer '{layer_name}'."
                )

            num_heads = tensor.shape[1]
            invalid_heads = [h for h in self.heads_to_keep if h < -num_heads or h >= num_heads]
            if invalid_heads:
                raise IndexError(
                    f"Head indices {invalid_heads} in heads_to_keep are out of bounds. "
                    f"Layer '{layer_name}' only has {num_heads} heads."
                )
            projected_attention[layer_name] = tensor[:, self.heads_to_keep, :, :]

        return projected_attention
