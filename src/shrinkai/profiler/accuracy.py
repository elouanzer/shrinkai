import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..utils import resolve_device


def compute_accuracy(model: nn.Module, dataloader: DataLoader, device: str | torch.device) -> float:
    """Computes the accuracy of a model over a validation dataloader.

    Handles both 2D logits (classification) and 3D logits (causal language modeling).

    Args:
        model: The standard PyTorch model.
        dataloader: Dataloader for evaluation after each epoch.
        device: Computing target ('auto', 'mps', 'cuda', 'cpu' or torch.device)
    Returns:
        float: Model's accuracy.
    """
    resolved_device = resolve_device(device)
    model.eval()
    model.to(resolved_device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].to(resolved_device), batch[1].to(resolved_device)
            b_size = inputs.size(0)

            outputs = model(inputs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            if logits.dim() == 3:
                # Causal LM shift
                preds = torch.argmax(logits[..., :-1, :].contiguous(), dim=-1)
                labels = labels[..., 1:].contiguous()
                total += labels.numel()
            else:
                # Standard classification
                preds = torch.argmax(logits, dim=-1)
                total += b_size

            correct += (preds == labels).sum().item()

    return (correct / total) * 100.0 if total > 0 else 0.0
