import torch
from torch import nn
import torch.nn.functional as F

class HybridEstimatorQNN(nn.Module):
    """
    Classical feed‑forward regressor with a pluggable quantum layer.

    The network mirrors the original EstimatorQNN but adds a
    placeholder ``_quantum_layer`` that can be overridden by a quantum
    implementation.  In the default pure‑classical mode this layer is a
    no‑op, allowing the model to be trained and evaluated without any
    quantum backend.
    """

    def __init__(self, input_dim: int = 2, hidden_sizes: tuple[int, int] = (8, 4), output_dim: int = 1):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, hidden_sizes[0])
        self.hidden2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.out = nn.Linear(hidden_sizes[1], output_dim)
        self._quantum_layer = lambda x: x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = F.tanh(self.hidden1(x))
        h2 = F.tanh(self.hidden2(h1))
        q = self._quantum_layer(h2)
        return self.out(q)

__all__ = ["HybridEstimatorQNN"]
