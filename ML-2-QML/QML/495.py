"""Quantum self‑attention using PennyLane variational circuits and expectation‑value attention weights."""

from __future__ import annotations

import pennylane as qml
import numpy as np
import torch
from pennylane import numpy as pnp


class SelfAttention:
    """
    Hybrid quantum‑classical self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be even; each pair encodes a token).
    device_name : str, optional
        PennyLane device to use. Defaults to 'qiskit.ibmq.qasm_simulator'.
    """

    def __init__(self, n_qubits: int = 8, device_name: str = "default.qubit"):
        if n_qubits % 2!= 0:
            raise ValueError("n_qubits must be even to pair tokens for attention.")
        self.n_qubits = n_qubits
        self.n_tokens = n_qubits // 2
        self.dev = qml.device(device_name, wires=n_qubits)

        # Learnable rotation parameters for each token
        self.rotation_params = torch.nn.Parameter(
            torch.randn(self.n_tokens, 3, dtype=torch.float64)
        )
        # Learnable entanglement parameters between adjacent token pairs
        self.entangle_params = torch.nn.Parameter(
            torch.randn(self.n_tokens - 1, dtype=torch.float64)
        )

    def _circuit(self, inputs: np.ndarray):
        """
        Quantum circuit that encodes inputs and applies parameterized rotations
        and controlled rotations to generate attention weights.
        """
        # Encode each token into a pair of qubits using Ry rotations
        for i in range(self.n_tokens):
            q1, q2 = 2 * i, 2 * i + 1
            qml.RY(inputs[i], wires=q1)
            qml.RY(inputs[i], wires=q2)

        # Parameterized single‑qubit rotations
        for i in range(self.n_tokens):
            qml.RX(self.rotation_params[i, 0], wires=2 * i)
            qml.RY(self.rotation_params[i, 1], wires=2 * i)
            qml.RZ(self.rotation_params[i, 2], wires=2 * i)

        # Controlled rotations for entanglement
        for i in range(self.n_tokens - 1):
            qml.CRX(self.entangle_params[i], wires=[2 * i, 2 * (i + 1)])

        # Measure expectation values of Z on each first qubit of token pairs
        return [qml.expval(qml.PauliZ(i)) for i in range(0, self.n_qubits, 2)]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute attention‑weighted sum of inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Tensor of shape (batch, n_tokens).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, n_tokens) after attention.
        """
        batch_size = inputs.shape[0]
        outputs = []

        for b in range(batch_size):
            # Run circuit on a single example
            @qml.qnode(self.dev, interface="torch")
            def circuit():
                return self._circuit(inputs[b].numpy())

            # Get raw expectation values
            raw = circuit()
            # Convert to attention weights using softmax
            attn = torch.softmax(raw, dim=0)
            # Weighted sum of the original inputs
            weighted = attn * inputs[b]
            outputs.append(weighted)

        return torch.stack(outputs)

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs)


__all__ = ["SelfAttention"]
