from __future__ import annotations

import torch
from torch import nn
from.qml_estimator import QuantumEstimator

class EstimatorQNN(nn.Module):
    """
    Hybrid estimator that combines a classical feed‑forward network with a
    parameterised quantum circuit.  The classical network is composed of
    Tanh‑activated linear layers followed by a scaling/shift block, mirroring
    the fraud‑detection construction.  The quantum part contributes a single
    expectation value that is fused with the classical output.
    """
    def __init__(
        self,
        n_features: int = 2,
        hidden_sizes: tuple[int,...] = (8, 4),
        scale_bounds: float = 5.0,
    ) -> None:
        super().__init__()
        # Classical feature extractor built from fraud‑detection style layers
        layers = []
        in_dim = n_features
        for h in hidden_sizes:
            lin = nn.Linear(in_dim, h, bias=True)
            # initialise weights within reasonable bounds
            nn.init.uniform_(lin.weight, -1.0, 1.0)
            nn.init.zeros_(lin.bias)
            layers.append(lin)
            layers.append(nn.Tanh())
            in_dim = h
        self.classical = nn.Sequential(*layers)

        # Quantum sub‑module
        self.quantum = QuantumEstimator()

        # Fusion layer
        self.fusion = nn.Linear(in_dim + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (..., n_features).  The first feature is
            passed to the quantum circuit as the input parameter; the second
            is treated as a weight parameter for the quantum rotation.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (..., 1).
        """
        c = self.classical(x)
        # Prepare tensor for quantum evaluation: only first two features
        q_input = x[..., :2]
        q = self.quantum(q_input)
        fused = torch.cat([c, q], dim=-1)
        return self.fusion(fused)

__all__ = ["EstimatorQNN"]
