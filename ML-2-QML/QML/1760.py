"""
Hybrid quantum‑classical regressor using Pennylane.
Replaces the toy one‑qubit circuit with a variational circuit that supports
parameter‑shift gradient evaluation and can be trained end‑to‑end.
"""

from __future__ import annotations

import pennylane as qml
import torch
import numpy as np

# Device for the quantum circuit
dev = qml.device("default.qubit", wires=1)

def _quantum_circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Simple variational circuit: H → RY(x) → RX(w) → measure Y."""
    qml.Hadamard(wires=0)
    qml.RY(inputs[0], wires=0)
    qml.RX(weights[0], wires=0)
    return qml.expval(qml.Y(0))

@qml.qnode(dev, interface="torch", diff_method="parameter-shift")
def quantum_node(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Quantum node that returns a scalar expectation value."""
    return _quantum_circuit(inputs.detach().cpu().numpy(),
                            weights.detach().cpu().numpy())

class EstimatorQNN:
    """
    Hybrid estimator that wraps a Pennylane variational circuit.
    Parameters
    ----------
    n_params : int, default 1
        Number of trainable quantum parameters (weights).
    """
    def __init__(self, n_params: int = 1) -> None:
        self.n_params = n_params
        # Initialise quantum weights as a torch Parameter
        self.weights = torch.nn.Parameter(torch.randn(n_params))
        self.device = torch.device("cpu")

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: returns the quantum expectation as a prediction.
        Parameters
        ----------
        inputs : torch.Tensor
            Input features of shape (..., 1)
        Returns
        -------
        torch.Tensor
            Predicted scalar output, shape (..., 1)
        """
        # Expand inputs to match required shape
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(-1)
        return quantum_node(inputs, self.weights).unsqueeze(-1)

__all__ = ["EstimatorQNN"]
