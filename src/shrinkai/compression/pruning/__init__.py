"""Pruning: inducing sparsity or physically shrinking a model.

Two complementary tools:

* `Pruner` (configured via `PruningConfig`) applies unstructured or structured
  masking through `torch.nn.utils.prune`. It never changes tensor shapes — even
  "structured" pruning here only zeroes out channels, so it does not by itself
  reduce parameter count, disk size, or latency on standard hardware.
* `ChannelPruner` physically removes pruned channels from `Conv2d`/`Linear`
  layers (and their dependent `BatchNorm`), using `torch.fx` to trace the
  model's dataflow graph and safely propagate the shrink downstream. This is
  what actually reduces parameters, size, latency, and FLOPs — but only on
  simple, non-branching topologies (see its own docstring for the exact scope).
"""

from .channel_pruner import ChannelPruner
from .pruning import Pruner, PruningConfig

__all__ = ["PruningConfig", "Pruner", "ChannelPruner"]
