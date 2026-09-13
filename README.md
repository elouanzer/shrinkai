<p align="center">
  <img src="https://raw.githubusercontent.com/elouanzer/shrinkai/main/docs/assets/shrinkai-logo-nobackground.png" alt="ShrinkAI Logo" width="200" />
</p>

[![PyPI version](https://img.shields.io/pypi/v/shrinkai?color=2860BE&style=flat-square&cacheSeconds=3600)](https://pypi.org/project/shrinkai/)
![Python versions](https://img.shields.io/pypi/pyversions/shrinkai?color=1595BC&style=flat-square&cacheSeconds=3600)
[![Documentation](https://img.shields.io/badge/docs-online-2DADCF?style=flat-square)](https://elouanzer.github.io/shrinkai/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-151922?logo=github&style=flat-square)](https://github.com/elouanzer/shrinkai)
![License](https://img.shields.io/github/license/elouanzer/shrinkai?color=7B8494&style=flat-square)

# ShrinkAI

**ShrinkAI** is a package for reducing neural networks size, making them ideal to run on small devices such as smartphones or robots, and for speeding up inference, particularly interesting for edge AI. This package includes, among other things, numerous distillation and compression techniques, all wrapped in an API that is easy to use for users familiar with PyTorch.

## Installation

ShrinkAI is on pypi and can be installed with the following command:
```bash
pip install shrinkai
```

The package is compatible with Python 3.11+, and depends on `torch`, `torchvision`, `rich`, `tqdm`, and `psutil`. Exporting to ONNX additionally requires the `export` extra:
```bash
pip install shrinkai[export]
```

## Quick Start

Distilling a smaller student model from a larger teacher only takes a `Distiller` and a loss:

```python
from shrinkai.distillation import Distiller
from shrinkai.distillation.losses import HintonLoss

# teacher, student: your PyTorch models. train_loader: your DataLoader.
distiller = Distiller(
    teacher=teacher,
    student=student,
    criterion=HintonLoss(),
    optimizer="adamw",
)

distiller.fit(train_dataloader=train_loader, epochs=10)
distiller.save_student("student.pt")
```

`shrinkai` also covers pruning, quantization, benchmarking, and exporting the trained model for deployment (ONNX, TorchScript). See the [full documentation](https://elouanzer.github.io/shrinkai/) for the complete quick start, tutorials, and API reference.

## Architecture

There is no flat top-level API: everything is imported from its submodule, e.g. `from shrinkai.distillation import Distiller`.

| Module | What it provides |
|---|---|
| `shrinkai.distillation` | `Distiller` (facade), `DistillationEngine`, training callbacks, and a library of losses under `shrinkai.distillation.losses` |
| `shrinkai.adapters` | `FeatureExtractor` (hook-based intermediate activations) and dimension-matching projectors |
| `shrinkai.analysis` | Representation alignment metrics between teacher and student (CKA, RSA, Spatial Attention) |
| `shrinkai.compression` | `Pruner` / `ChannelPruner` (mask-based and physical pruning) and `Quantizer` (PTQ/QAT) |
| `shrinkai.profiler` | `Profiler`, `count_flops`, latency and memory measurement |
| `shrinkai.export` | `export_onnx`, `export_torchscript` for deployment outside PyTorch |

## Examples

The [`docs/tutorials/`](docs/tutorials) folder has six runnable notebooks, from a basic distillation walkthrough to full deployment:

| # | Notebook | Covers |
|---|---|---|
| 01 | [Distillation on CIFAR10](docs/tutorials/01_distillation_vision_cifar10.ipynb) | `HintonLoss`, `FeatureLoss`, combining losses |
| 02 | [Distillation on SST-2](docs/tutorials/02_distillation_nlp_sst2.ipynb) | Adapting HuggingFace models, `AttentionMapLoss` |
| 03 | [LLM text generation](docs/tutorials/03_distillation_llm_generation.ipynb) | `ReverseKLLoss`, mismatched tokenizers, sequence-level KD |
| 04 | [Compression](docs/tutorials/04_compression.ipynb) | `Pruner`, `ChannelPruner`, `Quantizer` |
| 05 | [Full training control](docs/tutorials/05_deep_training.ipynb) | Custom losses, callbacks, AMP, checkpointing, custom engines |
| 06 | [Exporting for deployment](docs/tutorials/06_export.ipynb) | `export_onnx`, `export_torchscript` |


## Contact & Contributing

You can report an [issue](https://github.com/elouanzer/shrinkai/issues) directly on GitHub. Bug reports, feature requests, and questions are all welcome.

Contributions are welcome too, whether it's a bug fix, a new feature, or a documentation improvement:

1. Open an issue first for anything non-trivial, to discuss the approach before you start.
2. Fork the repository and work on a dedicated branch.
3. Add or update tests for any behavior change (run `uv run pytest` locally before opening a PR).
4. Keep the code clean: `uv run ruff check` and `uv run ruff format`.
5. Open a pull request against `main`, CI runs the test suite, coverage, and lint checks automatically.

## Citation

If you use ShrinkAI in your work and think it was helpful, please cite it as:

```bibtex
@software{shrinkai2026,
  author  = {Elouan Marsot},
  title   = {ShrinkAI},
  url     = {https://github.com/elouanzer/shrinkai},
  license = {MIT},
  version = {0.1.0}
}
```

## License

[MIT License](https://github.com/elouanzer/shrinkai/blob/main/LICENSE)
