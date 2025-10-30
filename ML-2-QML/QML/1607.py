"""
Hybrid quantum‑classical model using Pennylane.
The architecture mirrors the classical pipeline but replaces the fully‑connected
layer with a variational quantum circuit that produces expectation values
which are then fed into a final linear head.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
from pennylane import numpy as np  # type: ignore


class QFCModel(nn.Module):
    """Quantum‑enhanced variant of QFCModel using Pennylane's variational ansatz."""

    def __init__(self, n_qubits: int = 4, n_classes: int = 4, device: str = "default.qubit") -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)
        self.n_params = n_qubits * 3 * 2  # 2 layers of strong entangling with 3 params each

        # Trainable parameters for the ansatz
        self.params = nn.Parameter(torch.randn(self.n_params))

        # QNode with Torch interface for autograd
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch", diff_method="backprop")

        # Classical head
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, n_classes),
            nn.BatchNorm1d(n_classes),
        )

    def _circuit(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        Quantum circuit: amplitude encoding followed by a strongly entangling ansatz.

        Parameters
        ----------
        x : torch.Tensor
            Feature vector of shape (n_qubits,) or (batch, n_qubits).
        params : torch.Tensor
            Variational parameters of shape (n_qubits, 3, 2).

        Returns
        -------
        torch.Tensor
            Expectation values of Pauli‑Z on each qubit.
        """
        # Amplitude encoding via rotation angles
        qml.templates.embeddings.AngleEmbedding(x, wires=range(self.n_qubits))
        # Strongly entangling layers
        qml.templates.layers.StronglyEntanglingLayers(params, wires=range(self.n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Normalized logits of shape (batch, n_classes).
        """
        bsz = x.shape[0]
        # Flatten to a 1‑D vector per sample
        flat = x.view(bsz, -1)
        # Truncate or pad to match the number of qubits
        if flat.shape[1] > self.n_qubits:
            flat = flat[:, : self.n_qubits]
        elif flat.shape[1] < self.n_qubits:
            pad = torch.zeros(bsz, self.n_qubits - flat.shape[1], device=flat.device)
            flat = torch.cat([flat, pad], dim=1)

        # Compute quantum expectations for each sample
        q_expect = self.qnode(flat, self.params)  # shape (batch, n_qubits)

        # Classical classification head
        out = self.classifier(q_expect)
        return out


__all__ = ["QFCModel"]
