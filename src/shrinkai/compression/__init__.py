"""Model compression: pruning and quantization.

Two independent, composable families of techniques for shrinking a model after
(or during) training:

* **Pruning** (`shrinkai.compression.pruning`): induces sparsity (`Pruner`) or
  physically removes channels (`ChannelPruner`).
* **Quantization** (`shrinkai.compression.quantization`): reduces numerical
  precision of weights/activations (`Quantizer`), via PTQ or QAT.

Both can be combined with `shrinkai.distillation` (e.g. quantization-aware
training driven by a `Distiller`) or used standalone on any `nn.Module`.
"""

from .pruning import ChannelPruner, Pruner, PruningConfig
from .quantization import QuantConfig, Quantizer

__all__ = ["Pruner", "PruningConfig", "ChannelPruner", "Quantizer", "QuantConfig"]
