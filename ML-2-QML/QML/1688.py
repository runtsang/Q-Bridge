"""Variational quantum regressor using Pennylane."""

import pennylane as qml
import torch
from torch import nn
import numpy as np


class EstimatorQNN(nn.Module):
    """
    Quantum neural network that mirrors the classical estimator.
    Uses a 2‑qubit variational circuit with entanglement.
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_layers: int = 2,
        device_name: str = "default.qubit",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device(device_name, wires=n_qubits, shots=None)

        # Trainable weights: one rotation per qubit per layer
        self.weight_params = nn.Parameter(
            torch.randn(n_qubits * n_layers * 3, dtype=dtype)
        )  # 3 rotations per qubit

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # Encode inputs: map [-1, 1] to [-π, π]
            for i, w in enumerate(inputs):
                qml.RY(w * np.pi, wires=i)

            # Variational layers
            offset = 0
            for _ in range(n_layers):
                for i in range(n_qubits):
                    qml.RX(weights[offset + i * 3 + 0], wires=i)
                    qml.RZ(weights[offset + i * 3 + 1], wires=i)
                    qml.RY(weights[offset + i * 3 + 2], wires=i)
                # Entanglement
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                offset += n_qubits * 3

            # Observable: Pauli‑Z on first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Computes the expectation value for each sample in ``inputs``.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, n_qubits) with values in [-1, 1].

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 1) with the predicted regression output.
        """
        batch = inputs.shape[0]
        # Ensure inputs are on the same device as weights
        inputs = inputs.to(self.weight_params.device)
        # Compute outputs
        out = torch.stack(
            [self.circuit(inputs[i], self.weight_params) for i in range(batch)],
            dim=0,
        )
        return out.unsqueeze(-1)


__all__ = ["EstimatorQNN"]
