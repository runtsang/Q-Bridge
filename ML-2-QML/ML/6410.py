"""Hybrid estimator that fuses classical feed‑forward regression with quantum self‑attention.

The model first projects the input through a variational quantum circuit that implements a
self‑attention style block.  The resulting expectation values are concatenated with the
raw inputs and fed into a small neural network.  This illustrates how quantum feature
engineering can be coupled to a purely classical learner.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np

# Import the quantum sub‑module (assumes it lives in the same package)
from.quantum_self_attention import QuantumSelfAttention


class HybridEstimatorQNN(nn.Module):
    """
    Classical neural network that uses a quantum self‑attention block
    to generate additional features.

    Parameters
    ----------
    input_dim : int
        Dimension of the classical input vector.
    hidden_dim : int
        Size of the hidden layer in the feed‑forward network.
    output_dim : int
        Dimension of the regression output (default 1).
    n_qubits : int
        Number of qubits used in the quantum self‑attention circuit.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 8,
        output_dim: int = 1,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.q_attention = QuantumSelfAttention(n_qubits=n_qubits)
        self.hidden = nn.Sequential(
            nn.Linear(input_dim + n_qubits, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, input_dim)``.

        Returns
        -------
        torch.Tensor
            Regression output of shape ``(batch, output_dim)``.
        """
        # Convert to numpy for the quantum part
        x_np = x.detach().cpu().numpy()
        batch_size = x_np.shape[0]
        quantum_features = []

        for i in range(batch_size):
            inp = x_np[i]
            # Map the input to rotation parameters: repeat each element
            # so that we have 3 * n_qubits parameters.
            needed = 3 * self.q_attention.n_qubits
            rot = np.tile(inp, needed // len(inp))
            if len(rot) < needed:
                rot = np.concatenate([rot, np.zeros(needed - len(rot))])
            # Entanglement parameters are set to zero for this demo.
            ent = np.zeros(self.q_attention.n_qubits - 1)
            qfeat = self.q_attention.run(rot, ent)
            quantum_features.append(qfeat)

        qfeat_tensor = torch.tensor(
            np.stack(quantum_features), dtype=torch.float32, device=x.device
        )
        combined = torch.cat([x, qfeat_tensor], dim=1)
        return self.hidden(combined)


def EstimatorQNN() -> HybridEstimatorQNN:
    """Return a hybrid estimator that combines quantum and classical components."""
    return HybridEstimatorQNN()


__all__ = ["EstimatorQNN", "HybridEstimatorQNN"]
