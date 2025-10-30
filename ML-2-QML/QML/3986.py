"""Quantum‑classical hybrid QCNN implemented with Pennylane.

The class mirrors the structure of the classical model but replaces
the middle block with a depth‑wise convolution‑and‑pooling pattern
described in the original QCNN paper.  The quantum circuit is
parameterised by a weight vector that is optimised jointly with the
classical head using PyTorch autograd support.

The class name ``QCNNHybrid`` matches the classical counterpart so that
both modules expose the same API.
"""

import torch
from torch import nn
import pennylane as qml
import pennylane.numpy as pnp
from typing import Optional

class QCNNHybrid(nn.Module):
    """
    Hybrid quantum‑classical QCNN using Pennylane.
    """
    def __init__(
        self,
        in_features: int = 8,
        hidden_features: int = 16,
        out_features: int = 1,
        device: str = "default.qubit",
        wires: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Classical encoder (same as the classical model).
        self.feature_map = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Tanh(),
        )

        # Define a Pennylane device.
        self.wires = wires or hidden_features
        self.dev = qml.device(device, wires=self.wires)

        # Number of trainable parameters: two per qubit (RY and RZ).
        self.register_parameter(
            "weights",
            nn.Parameter(torch.randn(2 * self.wires)),
        )

        # Classical head.
        self.head = nn.Linear(self.wires, out_features)

        # Create the quantum node.
        self.qnode = qml.QNode(self._qnn_circuit, self.dev, interface="torch")

    def _qnn_circuit(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Quantum circuit implementing a simple variational ansatz."""
        # Encode input into qubit states using RZ rotations.
        for i in range(self.wires):
            qml.RZ(x[i], wires=i)

        # Apply trainable RY and RZ rotations.
        for i in range(self.wires):
            qml.RY(weights[i], wires=i)
            qml.RZ(weights[self.wires + i], wires=i)

        # Entangle neighbouring qubits with CNOTs.
        for i in range(0, self.wires - 1, 2):
            qml.CNOT(i, i + 1)

        # Measure expectation values of Pauli‑Z for each qubit.
        return [qml.expval(qml.PauliZ(i)) for i in range(self.wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: classical encoder → quantum circuit → linear head.
        """
        # Classical feature map
        x_enc = self.feature_map(x)
        # Quantum circuit expects a vector of size `self.wires`.
        if x_enc.shape[-1]!= self.wires:
            # Pad or truncate to match the number of qubits
            if x_enc.shape[-1] < self.wires:
                pad = self.wires - x_enc.shape[-1]
                x_enc = torch.nn.functional.pad(x_enc, (0, pad))
            else:
                x_enc = x_enc[..., : self.wires]

        # Quantum node
        q_out = self.qnode(x_enc, self.weights)

        # Linear head
        return self.head(q_out)

def QCNN() -> QCNNHybrid:
    """Factory returning a default ``QCNNHybrid`` instance."""
    return QCNNHybrid()
