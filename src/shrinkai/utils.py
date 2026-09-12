import torch


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolves the computation device automatically.

    Args:
        device: Computing target ('auto', 'mps', 'cuda', 'cpu' or torch.device).
            Defaults to 'auto'.
    Returns:
        torch.device: torch device."""
    if isinstance(device, str):
        if device == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            return torch.device(device)
    else:
        return device
