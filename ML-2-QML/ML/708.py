import torch
from torch import nn
import numpy as np
from typing import Iterable, Optional

class FCL(nn.Module):
    """
    Enhanced fully connected layer supporting multiple output units,
    optional bias, activation, dropout and batch input.
    """
    def __init__(self,
                 n_features: int,
                 n_outputs: int = 1,
                 activation: Optional[nn.Module] = None,
                 dropout: float = 0.0,
                 bias: bool = True,
                 device: str = "cpu"):
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs, bias=bias).to(device)
        self.activation = activation or nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.device = device

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass accepting batched inputs.
        """
        x = self.linear(inputs.to(self.device))
        x = self.activation(x)
        x = self.dropout(x)
        return x

    def run(self, thetas: Iterable[float], batch: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Mimic the original API: replace the linear weights with ``thetas``,
        run a forward pass on ``batch`` (if provided) or a dummy vector.
        Returns a NumPy array of the mean activation.
        """
        # Reshape thetas into weight matrix
        weight_shape = self.linear.weight.shape
        bias_shape = self.linear.bias.shape if self.linear.bias is not None else (0,)
        weight_size = weight_shape[0] * weight_shape[1]
        bias_size = bias_shape[0]
        total = weight_size + bias_size
        theta_arr = np.asarray(list(thetas), dtype=np.float32)

        if theta_arr.size!= total:
            raise ValueError(f"Expected {total} parameters, got {theta_arr.size}")

        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(theta_arr[:weight_size].reshape(weight_shape), dtype=torch.float32))
            if bias_size > 0:
                self.linear.bias.copy_(torch.tensor(theta_arr[weight_size:], dtype=torch.float32))

        # Default batch if none provided
        if batch is None:
            batch = torch.randn(10, self.linear.in_features, device=self.device)

        out = self.forward(batch)
        return out.detach().cpu().numpy().mean(axis=0)

__all__ = ["FCL"]
