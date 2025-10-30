"""Hybrid quantum‑classical regressor using Pennylane.

The quantum part is a variational circuit that embeds classical features
into a 2‑qubit system and measures the expectation value of Pauli‑Y.
Classical preprocessing is performed with a small neural network.
"""

import pennylane as qml
import torch
import numpy as np

class EstimatorQNN:
    """
    Hybrid quantum‑classical regression model.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used in the variational circuit (default 2).
    n_layers : int
        Depth of the variational ansatz (default 2).
    """
    def __init__(self, n_qubits: int = 2, n_layers: int = 2) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Classical embedding network
        self.classical = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, n_qubits),
        )

        # Learnable quantum parameters
        self.weight_params = torch.nn.Parameter(
            torch.randn(n_layers, n_qubits, dtype=torch.float64)
        )

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weight_params: torch.Tensor) -> torch.Tensor:
            # Encode classical features into qubit rotations
            for i, wire in enumerate(range(n_qubits)):
                qml.RX(inputs[i], wires=wire)
            # Variational layers
            for layer in range(n_layers):
                for wire in range(n_qubits):
                    qml.RZ(weight_params[layer, wire], wires=wire)
                for wire in range(n_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])
            # Return expectation value of Pauli‑Y on the first qubit
            return qml.expval(qml.PauliY(0))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, 2) containing the raw features.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch,) containing the regression outputs.
        """
        # Classical embedding
        embedded = self.classical(inputs)
        # Quantum circuit evaluation
        return self.circuit(embedded, self.weight_params)

def estimator_qnn() -> EstimatorQNN:
    """Return an instance of the hybrid EstimatorQNN model."""
    return EstimatorQNN()

__all__ = ["EstimatorQNN", "estimator_qnn"]
