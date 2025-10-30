"""EstimatorQNNExtendedQML – a hybrid variational circuit with Pennylane.

The QNode is defined as a callable that accepts classical inputs and
weight parameters and returns the expectation value of a Pauli‑Z
observable.  It is wrapped in a PyTorch nn.Module so gradients flow
through the quantum circuit during optimisation.
"""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
import torch
from torch import nn
from typing import Tuple


class EstimatorQNNExtendedQML(nn.Module):
    """
    Hybrid quantum neural network using Pennylane's automatic
    differentiation.  The circuit consists of layers of
    `ry` and `rz` rotations followed by a full‑to‑full entangling
    pattern.  The expectation of `qml.PauliZ(0)` is returned as the
    network output.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 2) -> None:
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit.
        n_layers : int
            Number of variational layers.
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Device – use the default local simulator
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Create the QNode; parameters are concatenated into one vector
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode inputs as Ry rotations on each qubit
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            for layer in range(n_layers):
                # Parameterised rotations
                for i in range(n_qubits):
                    qml.RZ(weights[layer * n_qubits + i], wires=i)
                # Entangling layer – full‑to‑full
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        qml.CNOT(wires=[i, j])

            # Measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        # Initialise weight parameters
        self.weight_shape = (n_layers * n_qubits,)
        self.weight_params = nn.Parameter(
            torch.randn(self.weight_shape, requires_grad=True)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the expectation value for a batch of inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch_size, n_qubits).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, 1) containing the network output.
        """
        batch_size = inputs.shape[0]
        outputs = []
        for i in range(batch_size):
            out = self.circuit(inputs[i], self.weight_params)
            outputs.append(out)
        return torch.stack(outputs).unsqueeze(-1)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Convenience wrapper for inference – no gradient tracking.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch_size, n_qubits).

        Returns
        -------
        torch.Tensor
            Prediction tensor.
        """
        with torch.no_grad():
            return self.forward(inputs)

__all__ = ["EstimatorQNNExtendedQML"]
