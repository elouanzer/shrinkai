"""ShrinkAI: knowledge distillation and model compression for PyTorch.

ShrinkAI provides an easy-to-use, PyTorch-native toolkit for making neural
networks smaller and faster: knowledge distillation (`shrinkai.distillation`),
pruning and quantization (`shrinkai.compression`), representation analysis
(`shrinkai.analysis`), profiling (`shrinkai.profiler`), and deployment export
(`shrinkai.export`).

There is no flat top-level API: import what you need from the relevant
submodule, e.g. `from shrinkai.distillation import Distiller`. See the API
Reference for the full list of submodules.
"""

__version__ = "0.1.0"
