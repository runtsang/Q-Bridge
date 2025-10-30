"""Hybrid quantum‑classical regressor using Pennylane.

The network consists of a variational circuit that accepts a batch
of real‑valued inputs encoded as rotation angles.  The circuit
produces a single expectation value per input, which is then
treated as a feature by the surrounding classical head.  The
model is fully differentiable through Pennylane's PyTorch
interface, enabling end‑to‑end training.
"""

import pennylane as qml
import torch
from torch import nn
from typing import Tuple

class EstimatorQNN(nn.Module):
    """
    A quantum neural network that maps real‑valued inputs to a
    single expectation value using a parameter‑shift circuit.

    Parameters
    ----------
    n_qubits : int, default 1
        Number of qubits in the variational circuit.
    n_layers : int, default 1
        Number of variational layers.
    device_name : str, default "default.qubit"
        Pennylane device name (e.g. "qiskit.ibmq_qasm_simulator").
    shots : int, default 1024
        Number of shots for expectation estimation.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        n_layers: int = 1,
        device_name: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device_name, wires=n_qubits, shots=shots)

        # Weight parameters for the variational layers
        self.weight_params = nn.Parameter(
            torch.randn(n_layers, n_qubits, requires_grad=True)
        )

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Feature encoding: rotate each qubit by the input value
            for i, x in enumerate(inputs):
                qml.RY(x, wires=i)
            # Variational layers
            for l in range(n_layers):
                for q in range(n_qubits):
                    qml.RZ(weights[l, q], wires=q)
                if n_qubits > 1:
                    # Simple entangling pattern (CNOT chain)
                    for q in range(n_qubits - 1):
                        qml.CNOT(wires=[q, q + 1])
            # Expectation value of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute the expectation value for each batch
        element and return a column vector suitable for concatenation
        with classical features.
        """
        # Ensure input shape (batch, n_qubits)
        if x.shape[-1]!= self.n_qubits:
            raise ValueError(f"Expected last dimension to be {self.n_qubits}")
        return self.circuit(x, self.weight_params).unsqueeze(-1)
