"""Hybrid classical fully‑connected layer with optional quantum augmentation.

This module defines the FullyConnectedLayer class that
provides a standard linear layer with a tanh activation.
Additionally it can compute a quantum‑augmented forward pass
via a variational circuit supplied by the QML module.
"""

import torch
from torch import nn
import numpy as np

class FullyConnectedLayer(nn.Module):
    """Fully connected layer with optional quantum augmentation.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_outputs : int, default=1
        Number of output units.
    dropout : float, default=0.0
        Dropout probability for regularization.
    """

    def __init__(self, n_features: int, n_outputs: int = 1, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.activation(self.dropout(self.linear(x)))

    def quantum_forward(self, thetas: np.ndarray, qml_module) -> np.ndarray:
        """
        Compute a quantum‑augmented forward pass.

        Parameters
        ----------
        thetas : np.ndarray
            Array of parameters for the variational circuit.
        qml_module : module
            Module containing a QuantumCircuit class.

        Returns
        -------
        np.ndarray
            Expectation value from the variational circuit.
        """
        circuit = qml_module.QuantumCircuit(1, shots=1000)
        return circuit.run(thetas)

    def state_dict(self, *args, **kwargs):
        """Return the state dict of the underlying linear layer."""
        return super().state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Load state dict into the underlying linear layer."""
        super().load_state_dict(*args, **kwargs)
