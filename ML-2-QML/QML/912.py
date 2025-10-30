"""Quantum self‑attention inspired by transformer attention.

The implementation uses PennyLane to build a variational circuit that
encodes the input sequence, applies a stack of rotation and entanglement
layers, and measures the expectation values of Pauli‑Z on each qubit.
These expectation values are interpreted as attention weights and
produced in a tensor shape compatible with the classical counterpart.

Key features
------------
* Parameter‑efficient variational circuit with configurable depth.
* Hybrid forward pass that can be differentiated with PyTorch autograd.
* Supports batched inputs and returns a tensor of shape
  ``(batch, n_qubits)`` that can be interpreted as attention scores.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch


class SelfAttentionLayer:
    """Variational self‑attention quantum layer."""

    def __init__(
        self,
        n_qubits: int,
        num_layers: int = 2,
        device: str = "default.qubit",
    ):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=n_qubits)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(
            rotation_params: torch.Tensor,
            entangle_params: torch.Tensor,
            input_vec: torch.Tensor,
        ) -> torch.Tensor:
            """
            Parameters
            ----------
            rotation_params : torch.Tensor
                Shape ``(num_layers, n_qubits)`` – RX angles for each layer.
            entangle_params : torch.Tensor
                Shape ``(num_layers, n_qubits)`` – RY angles for each layer.
            input_vec : torch.Tensor
                Shape ``(n_qubits,)`` – angle‑encoded input data.
            """
            # Angle‑encoding of the input
            for i in range(self.n_qubits):
                qml.RY(input_vec[i], wires=i)

            # Variational layers
            for l in range(self.num_layers):
                for i in range(self.n_qubits):
                    qml.RX(rotation_params[l, i], wires=i)
                    qml.RY(entangle_params[l, i], wires=i)
                # Entangling layer (full‑chain + wrap‑around)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """
        Execute the variational circuit and return the expectation values.

        Parameters
        ----------
        rotation_params : np.ndarray
            Shape ``(num_layers, n_qubits)`` – RX angles.
        entangle_params : np.ndarray
            Shape ``(num_layers, n_qubits)`` – RY angles.
        inputs : np.ndarray
            Shape ``(batch, n_qubits)`` – input data.

        Returns
        -------
        np.ndarray
            Shape ``(batch, n_qubits)`` – expectation values interpreted as
            attention scores.
        """
        rot_t = torch.as_tensor(rotation_params, dtype=torch.float32)
        ent_t = torch.as_tensor(entangle_params, dtype=torch.float32)
        out = []

        for inp in inputs:
            inp_t = torch.as_tensor(inp, dtype=torch.float32)
            expvals = self.circuit(rot_t, ent_t, inp_t)
            out.append(expvals.detach().cpu().numpy())

        return np.stack(out, axis=0)


__all__ = ["SelfAttentionLayer"]
