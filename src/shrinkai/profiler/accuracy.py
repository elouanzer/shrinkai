import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_accuracy(model: nn.Module, dataloader: DataLoader, device: str | torch.device) -> float:
    """Computes the accuracy of a model over a validation dataloader.

    Handles both 2D logits (classification) and 3D logits (causal language modeling).
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
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
