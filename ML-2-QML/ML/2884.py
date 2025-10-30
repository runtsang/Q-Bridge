import torch
import numpy as np
from torch import nn
from typing import Iterable

class HybridFCLEstimator(nn.Module):
    """
    Classical hybrid model that mimics a quantum fully‑connected layer.
    The network consists of:
    - A classical feed‑forward network (mimicking EstimatorQNN) that
      transforms the input features into a set of weight parameters.
    - A quantum layer approximation implemented with a simple
      parameterised linear + tanh operation, which emulates the
      expectation value of a single‑qubit RY rotation followed by
      measurement.
    """

    def __init__(self, input_dim: int = 2, hidden_dim: int = 8) -> None:
        super().__init__()
        # Classical pre‑processing network
        self.pre_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classical network, producing a scalar
        that will be interpreted as a quantum expectation value.
        """
        return self.pre_net(x)

    def run(self, inputs: np.ndarray, thetas: Iterable[float]) -> np.ndarray:
        """
        End‑to‑end evaluation that accepts raw input features and a
        sequence of quantum parameters (thetas). The quantum parameters
        are ignored in this classical approximation; they are kept only
        to preserve API compatibility with the QML counterpart.
        """
        inp = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.forward(inp)
        # Emulate a single‑qubit expectation: tanh(linear(thetas))
        theta_vec = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        quantum_expectation = torch.tanh(theta_vec @ out.unsqueeze(1)).mean()
        return quantum_expectation.detach().numpy()

__all__ = ["HybridFCLEstimator"]
