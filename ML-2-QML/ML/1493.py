import numpy as np
import torch
from torch import nn
from typing import Iterable, Sequence

class FCL(nn.Module):
    """
    Enhanced fully connected layer with optional hidden layers and dropout.
    The `run` method mimics the quantum interface: it loads a list of
    parameters, performs a forward pass on a dummy input, and returns the
    mean output as a NumPy array.
    """

    def __init__(self,
                 n_features: int = 1,
                 hidden_sizes: Sequence[int] = (),
                 dropout: float = 0.0) -> None:
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.Tanh())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
        # Keep parameters in a predictable order for `run`
        self._param_tensors = list(self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Load the provided theta vector into the network's parameters,
        execute a forward pass on a synthetic input, and return the
        resulting output as a NumPy array.
        """
        # Ensure the number of supplied parameters matches the model's size
        total_params = sum(p.numel() for p in self._param_tensors)
        theta_list = list(thetas)
        if len(theta_list)!= total_params:
            raise ValueError(
                f"Expected {total_params} parameters, got {len(theta_list)}"
            )
        # Assign values
        offset = 0
        for p in self._param_tensors:
            num = p.numel()
            new_vals = torch.tensor(theta_list[offset:offset + num],
                                    dtype=p.dtype,
                                    device=p.device).view_as(p)
            p.data = new_vals
            offset += num
        # Dummy input with matching dimension
        dummy = torch.randn(1, self.model[0].in_features)
        out = self.forward(dummy)
        return out.detach().cpu().numpy()

__all__ = ["FCL"]
