import torch
from torch import nn
import numpy as np

class FCL(nn.Module):
    """
    Advanced fully connected layer for classical experiments.
    Supports multiple output neurons, bias, dropout, and batch input.
    Mimics a quantum layer while remaining fully differentiable.
    """
    def __init__(self, in_features: int, out_features: int = 1,
                 bias: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.Tanh()

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        """
        Forward pass accepting a tensor of shape (batch, in_features) or a flat vector.
        Applies linear transform, tanh activation, and optional dropout.
        """
        if thetas.dim() == 1:
            thetas = thetas.unsqueeze(0)
        out = self.linear(thetas)
        out = self.activation(out)
        out = self.dropout(out)
        return out.squeeze()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Convenience wrapper: converts NumPy array to torch tensor, runs forward,
        and returns NumPy array. No gradients are computed.
        """
        with torch.no_grad():
            tensor = torch.as_tensor(thetas, dtype=torch.float32)
            return self.forward(tensor).detach().cpu().numpy()

__all__ = ["FCL"]
