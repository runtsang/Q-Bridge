"""Enhanced fully connected layer with optional dropout and batch support."""
import torch
from torch import nn
import numpy as np
from typing import Iterable, Optional, Callable

class FCL(nn.Module):
    """
    Classical fully connected layer with additional features:
    - Configurable number of output units.
    - Optional dropout for regularization.
    - Supports batch inputs and automatic differentiation.
    """
    def __init__(
        self,
        n_features: int,
        n_outputs: int = 1,
        dropout_prob: float = 0.0,
        bias: bool = True,
        init_fn: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs, bias=bias)
        self.dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else None
        if init_fn is not None:
            init_fn(self.linear.weight)
            if bias:
                init_fn(self.linear.bias)

    def forward(self, thetas: Iterable[float]) -> torch.Tensor:
        """
        Forward pass for a single sample or batch.
        ``thetas`` should be a 1‑D iterable of size ``n_features`` or ``(batch, n_features)``.
        Returns a tensor of shape ``(batch, n_outputs)``.
        """
        x = torch.tensor(thetas, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if self.dropout is not None:
            x = self.dropout(x)
        out = self.linear(x)
        return out

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compatibility wrapper: returns NumPy array of means over the batch.
        """
        with torch.no_grad():
            out = self.forward(thetas)
        return out.mean(dim=0).detach().numpy()
