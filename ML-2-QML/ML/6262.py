"""Hybrid classical‑quantum estimator combining linear layers with a
parameterized quantum circuit.

The public class `HybridEstimatorQNN` is a standard PyTorch
`nn.Module`.  It first transforms the input through two hidden
classical layers, then feeds the reduced activations into a quantum
layer that returns the expectation of a Pauli‑Z operator.  The
quantum output is finally mapped to the regression target.
"""

import torch
from torch import nn
from.quantum_layer import QuantumLayer

class HybridEstimatorQNN(nn.Module):
    """
    A hybrid estimator that stacks classical nonlinear transforms before
    a quantum layer.  The quantum layer implements a single‑qubit
    variational circuit whose parameters are derived from the classical
    activations.  The output of the quantum layer is then passed through
    a final linear mapping to produce the regression target.
    """
    def __init__(self, input_dim: int = 2, hidden_dim: int = 8, output_dim: int = 1):
        super().__init__()
        self.classical = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.quantum = QuantumLayer()
        self.final = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the hybrid network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch, output_dim).
        """
        # Classical feature extraction
        x = self.classical(x)          # (batch, hidden_dim//2)
        # Reduce to a single scalar per sample for the quantum layer
        q_input = x.mean(dim=1)        # (batch,)
        # Quantum expectation value
        q = self.quantum(q_input)      # (batch,)
        q = q.unsqueeze(-1)            # (batch, 1)
        # Final linear mapping
        return self.final(q)

__all__ = ["HybridEstimatorQNN"]
