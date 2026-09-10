import warnings
from collections import defaultdict

import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

from ..adapters import FeatureExtractor
from ..utils import resolve_device
from .metrics import FEATURE_METRICS, BaseFeatureMetric


class FeatureAnalyzerReport:
    """Holds representation analysis results and renders a dashboard."""

    def __init__(self, layer_scores: dict[str, dict[str, float]]) -> None:
        """Initializes the report.

        Args:
            layer_scores: Nested dict `{ "stage_1": {"CKA (Linear)": 0.85}, ... }`.
        """
        self.layer_scores = layer_scores

    def show(self) -> None:
        """Renders an interactive formatted analysis table in the console."""
        if not self.layer_scores:
            print("No data to report.")
            return

        console = Console()
        table = Table(
            title="Representation Alignment Report (Feature Distillation)",
            header_style="bold magenta",
        )

        table.add_column("Layer / Stage Alias", style="bold")

        first_layer = next(iter(self.layer_scores.values()))
        metric_names = list(first_layer.keys())
        for metric in metric_names:
            table.add_column(metric, justify="right")

        for layer_name, metrics in self.layer_scores.items():
            row_data = [layer_name]
            for metric in metric_names:
                score = metrics.get(metric, 0.0)
                color = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
                row_data.append(f"[{color}]{score:.3f}[/{color}]")
            table.add_row(*row_data)

        console.print(table)


class FeatureAnalyzer:
    """Evaluates how well a student model mimics the teacher's internal hidden states."""

    def __init__(
        self,
        teacher_extractor: FeatureExtractor,
        student_extractor: FeatureExtractor,
        device: torch.device | str = "auto",
    ) -> None:
        """Initializes the FeatureAnalyzer.

        Args:
            teacher_extractor: Teacher model wrapped in FeatureExtractor.
            student_extractor: Student model wrapped in FeatureExtractor.
            device: Computing target.
        """
        self.teacher = teacher_extractor
        self.student = student_extractor
        self.device = resolve_device(device)
        self.teacher.to(self.device)
        self.student.to(self.device)

    def evaluate(
        self,
        dataloader: DataLoader,
        metrics: list[str | BaseFeatureMetric] | None = None,
    ) -> FeatureAnalyzerReport:
        """Runs the dataset through both models and computes alignment metrics per layer.

        Args:
            dataloader: Dataloader yielding validation batches.
            metrics: List of metric strings ('cka') or instantiated BaseFeatureMetric objects.

        Returns:
            FeatureAnalyzerReport: Formatted report ready for `.show()`.
        """
        if metrics is None:
            metrics = ["cka"]
        self.teacher.eval()
        self.student.eval()

        active_metrics = []
        for m in metrics:
            if isinstance(m, str):
                if m.lower() not in FEATURE_METRICS:
                    raise ValueError(
                        f"Unknown metric '{m}'. Available: {list(FEATURE_METRICS.keys())}"
                    )
                active_metrics.append(FEATURE_METRICS[m.lower()])
            else:
                active_metrics.append(m)

        accumulated_scores = defaultdict(lambda: defaultdict(float))
        num_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs = batch[0].to(self.device)
                num_batches += 1

                _, s_features = self.student(inputs)
                _, t_features = self.teacher(inputs)

                common_keys = set(s_features.keys()).intersection(set(t_features.keys()))

                if batch_idx == 0 and not common_keys:
                    warnings.warn(
                        "No common alias keys between the teacher and the studen."
                        "Check `target_layers` in your FeatureExtractors.",
                        stacklevel=2,
                    )
                    break

                for key in common_keys:
                    for metric in active_metrics:
                        score = metric.compute(s_features[key], t_features[key])
                        accumulated_scores[key][metric.name] += score

        if num_batches == 0:
            raise ValueError("No batches in the provided dataloader.")

        final_scores = defaultdict(dict)
        for layer, metric_dict in accumulated_scores.items():
            for metric_name, total_score in metric_dict.items():
                final_scores[layer][metric_name] = total_score / num_batches

        return FeatureAnalyzerReport(dict(final_scores))
