from __future__ import annotations

import numpy as np
import torch
from torch import nn
from typing import Iterable

class HybridEstimatorQNN(nn.Module):
    """
    A hybrid regressor that extends the original EstimatorQNN by adding
    an optional quantum feature map.  The classical trunk is a fully
    connected network; the quantum layer, if enabled, operates on the
    output of the last hidden unit and returns a scalar expectation.
    """

    def __init__(
        self,
        n_features: int = 2,
        hidden_sizes: Iterable[int] | None = None,
        use_quantum: bool = False,
        quantum_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [max(8, n_features * 2)]
        layers = []
        in_features = n_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_features, h))
            layers.append(nn.Tanh())
            in_features = h
        layers.append(nn.Linear(in_features, 1))
        self.trunk = nn.Sequential(*layers)

        self.use_quantum = use_quantum
        if use_quantum:
            if quantum_layer is None:
                # Identity quantum layer that simply passes the scalar forward
                self.quantum_layer = nn.Identity()
            else:
                self.quantum_layer = quantum_layer
        else:
            self.quantum_layer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.trunk(x)
        if self.use_quantum:
            out = self.quantum_layer(out)
        return out

    def run(self, x: np.ndarray) -> np.ndarray:
        """Convenience wrapper that accepts NumPy input and returns NumPy output."""
        tensor = torch.as_tensor(x, dtype=torch.float32)
        with torch.no_grad():
            out = self.forward(tensor)
        return out.cpu().numpy()

__all__ = ["HybridEstimatorQNN"]
