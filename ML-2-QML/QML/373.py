"""Quantum quanvolution module using Pennylane variational circuits.

This implementation replaces the fixed random layer from the seed with
a parameterised ansatz that can be trained jointly with the classifier.
Each 2×2 image patch is encoded via Ry rotations, followed by two
entangling layers of CNOTs and a layer of trainable Rz rotations.
The measurement is the expectation value of the Pauli‑Z operator
on all wires, producing a 4‑dimensional feature vector per patch.

The class :class:`Quanvolution__gen413` mirrors the classical version
but inherits from :class:`torch.nn.Module` to expose a quantum
forward pass that can be differentiated with respect to the circuit
parameters.
"""

from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class Quanvolution__gen413(nn.Module):
    """Quantum quanvolution filter with a variational ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits per patch (default 4 for a 2×2 patch).
    device_name : str
        Pennylane device name (e.g. 'default.qubit').
    """

    def __init__(
        self,
        num_qubits: int = 4,
        device_name: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.dev = qml.device(device_name, wires=num_qubits)

        # Variational parameters: 2 layers of Rz on each qubit
        self.params = nn.Parameter(
            torch.randn(2, num_qubits)
        )  # shape (layer, qubit)

        # Classical head: MLP similar to the classical counterpart
        self.mlp = nn.Sequential(
            nn.Linear(4 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 10),
        )

    def _quantum_block(self, x_patch: torch.Tensor) -> torch.Tensor:
        """Apply the variational circuit to a single patch.

        Parameters
        ----------
        x_patch : torch.Tensor
            Tensor of shape (batch, 4) containing the 2×2 pixel values.

        Returns
        -------
        torch.Tensor
            Expectation values of Pauli‑Z on each qubit, shape (batch, 4).
        """

        @qml.qnode(self.dev, interface="torch")
        def circuit(p, x):
            # Encode the pixel values using Ry rotations
            for i in range(self.num_qubits):
                qml.RY(x[i], wires=i)
            # Entangling layer
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Variational layer
            for i in range(self.num_qubits):
                qml.RZ(p[0, i], wires=i)
            # Second entangling layer
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Second variational layer
            for i in range(self.num_qubits):
                qml.RZ(p[1, i], wires=i)
            # Measure expectation of Pauli‑Z for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return circuit(self.params, x_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the quantum filter and classical head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, 10).
        """
        bsz = x.size(0)
        # Flatten to 28x28 patches
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, 0, r : r + 2, c : c + 2]
                # Flatten patch to (batch, 4)
                patch_flat = patch.view(bsz, -1)
                out = self._quantum_block(patch_flat)
                patches.append(out)
        # Concatenate all patch features: shape (batch, 4*14*14)
        features = torch.cat(patches, dim=1)
        logits = self.mlp(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution__gen413"]
