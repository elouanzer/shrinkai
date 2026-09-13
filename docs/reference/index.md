# API Reference

This is the exhaustive, auto-generated reference for every public module of
ShrinkAI, browsable via the navigation tree on the left, drill into a
package to see its modules, and into a module to see its classes and
functions, complete with signatures and source.

There is no flat `import shrinkai` API: everything is imported from its
submodule, e.g. `from shrinkai.distillation import Distiller`. If you're
looking for a guided introduction instead of a reference, start with the
[Quickstart](../a_quick_start.ipynb) then the [Tutorials](../tutorials/01_distillation_vision_cifar10.ipynb).

## Where to start, by task

| I want to... | Start here |
|---|---|
| Distill a teacher into a smaller student | [`Distiller`](distillation/distiller.md) |
| Pick or write a distillation loss | [`shrinkai.distillation.losses`](distillation/losses/base.md) |
| Stop training early / save the best checkpoint | [`EarlyStopping`, `ModelCheckpoint`](distillation/callbacks.md) |
| Compare intermediate representations (teacher vs. student) | [`FeatureExtractor`](adapters/extractor.md), [`FeatureAnalyzer`](analysis/analyzer.md) |
| Reconcile mismatched feature dimensions | [`FeatureProjector`, `AttentionHeadSelector`](adapters/projector.md) |
| Prune a model | [`Pruner`](compression/pruning/pruning.md) (masking), [`ChannelPruner`](compression/pruning/channel_pruner.md) (physical shrink) |
| Quantize a model (PTQ / QAT) | [`Quantizer`](compression/quantization/quantization.md) |
| Measure latency, size, params, FLOPs | [`Profiler`, `count_flops`](profiler/benchmark.md) |
| Export a trained model for deployment | [`export_onnx`, `export_torchscript`](export/exporter.md) |

## Package map

<div class="grid cards" markdown>

- :material-brain: **[`shrinkai.distillation`](distillation/index.md)**

    Train a student to mimic a teacher: `Distiller` (high-level facade),
    `DistillationEngine` (training loop), ready-made callbacks, and a
    library of losses under `shrinkai.distillation.losses`.

- :material-link-variant: **[`shrinkai.adapters`](adapters/index.md)**

    Bridge teacher/student architectures: hook-based feature extraction and
    dimension-matching projectors.

- :material-magnify: **[`shrinkai.analysis`](analysis/index.md)**

    Score how well a student's internal representations align with its
    teacher's (CKA, RSA, spatial attention).

- :material-content-cut: **[`shrinkai.compression`](compression/index.md)**

    Shrink a model after (or during) training: pruning
    (`shrinkai.compression.pruning`) and quantization
    (`shrinkai.compression.quantization`).

- :material-speedometer: **[`shrinkai.profiler`](profiler/index.md)**

    Measure and compare parameter count, disk size, latency, FLOPs, and
    memory footprint.

- :material-export: **[`shrinkai.export`](export/index.md)**

    Get a trained/compressed model out of Python: ONNX and TorchScript export.

</div>
