import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Union

class FCL(nn.Module):
    """
    Extended fully‑connected layer with hidden units, ReLU, and dropout.
    Supports batched inputs and a convenient `run` wrapper that accepts
    iterables or tensors, mirroring the original seed interface.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.net(x)

    def run(self, thetas: Union[Iterable[float], torch.Tensor]) -> torch.Tensor:
        """
        Accepts a batch of input thetas either as an iterable of floats
        or as a torch.Tensor. Returns the layer output as a NumPy array.
        """
        if isinstance(thetas, (list, tuple)):
            thetas = torch.tensor(thetas, dtype=torch.float32).view(-1, 1)
        elif isinstance(thetas, torch.Tensor):
            thetas = thetas.view(-1, 1)
        else:
            raise TypeError("Input must be an iterable or torch.Tensor")
        with torch.no_grad():
            out = self.forward(thetas)
        return out.cpu().numpy()

__all__ = ["FCL"]
