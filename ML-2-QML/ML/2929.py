import torch
import torch.nn as nn
import numpy as np
from typing import Iterable, Tuple

class QuantumHybridClassifier(nn.Module):
    """
    Hybrid classifier that combines a classical feed‑forward network with a
    quantum‑inspired fully connected layer.  The quantum layer is implemented
    as a lightweight torch module that mimics the behaviour of the Qiskit
    FCL example in the seed.  This design allows the same API to be used for
    purely classical experiments or for hybrid experiments that later
    swap in a real quantum backend.
    """

    def __init__(self, num_features: int, depth: int, use_quantum_layer: bool = True):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.use_quantum_layer = use_quantum_layer

        # Classical backbone
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.classical = nn.Sequential(*layers)

        # Quantum‑inspired layer
        if use_quantum_layer:
            self.qcl = self._build_quantum_layer(num_features)
        else:
            self.qcl = None

    def _build_quantum_layer(self, n_features: int) -> nn.Module:
        """
        Return a torch module that reproduces the behaviour of the
        Qiskit FCL example.  The layer accepts a vector of parameters
        and returns a single expectation value.
        """
        class _QuantumLayer(nn.Module):
            def __init__(self, n: int):
                super().__init__()
                self.linear = nn.Linear(n, 1)

            def forward(self, thetas: Iterable[float]) -> torch.Tensor:
                values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
                exp = torch.tanh(self.linear(values)).mean(dim=0)
                return exp

        return _QuantumLayer(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classical backbone.  The quantum layer
        is not part of the differentiable path; it can be called
        separately via ``quantum_expectation``.
        """
        return self.classical(x)

    def quantum_expectation(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the expectation value of the quantum‑inspired layer.
        Returns a NumPy array to match the API of the Qiskit example.
        """
        if self.qcl is None:
            raise RuntimeError("Quantum layer not enabled.")
        with torch.no_grad():
            exp = self.qcl(list(thetas))
        return exp.detach().cpu().numpy()

__all__ = ["QuantumHybridClassifier"]
