import numpy as np
import torch
from torch import nn

class FCLGen(nn.Module):
    """
    Extended fully connected layer.

    Features:
    * Multi‑output support.
    * Optional dropout for regularisation.
    * `run` method retained for backward compatibility.
    * `forward` integrates with PyTorch autograd.
    """
    def __init__(self, n_features: int = 1, out_features: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, out_features)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        thetas : torch.Tensor
            Shape (batch, n_features). Values are treated as inputs to the linear layer.
        """
        x = self.linear(thetas)
        x = self.dropout(x)
        # use tanh non‑linearity as in the seed
        return torch.tanh(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compatibility wrapper that accepts an iterable of floats and returns a NumPy array.
        """
        tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        with torch.no_grad():
            out = self.forward(tensor).mean(dim=0)
        return out.detach().numpy()

__all__ = ["FCLGen"]
