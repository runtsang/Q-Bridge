"""Hybrid quantum model using Pennylane.

The model encodes classical features into a 4‑qubit quantum circuit.
The circuit consists of data‑encoding RY gates, a layer of parameter‑shared
RY/Z rotations, and a chain of CNOTs. The output is a 4‑dimensional vector
of Pauli‑Z expectation values, batch‑normalized and usable in a hybrid
training loop.
"""

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

# Default qubit device; replace with a real QPU device if desired
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(x_enc_batch: torch.Tensor, params: torch.Tensor):
    """Variational circuit applied to each batch element."""
    out = []
    for xb in x_enc_batch:
        # Data encoding: map 16‑dim vector to 4 RY gates
        for i in range(4):
            qml.RY(xb[i], wires=i)
        # Parameter‑shared rotation layer
        for i in range(4):
            qml.RZ(params[i, 0], wires=i)
            qml.RX(params[i, 1], wires=i)
        # Entangling layer
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        # Measure Pauli‑Z expectation values
        out.append([qml.expval(qml.PauliZ(i)) for i in range(4)])
    return torch.stack(out)

class QFCModel(nn.Module):
    """Hybrid quantum model with a classical encoder and a Pennylane variational layer."""

    def __init__(self):
        super().__init__()
        # Classical encoder: average pooling to reduce spatial dimension
        self.encoder = nn.AvgPool2d(kernel_size=6)
        # Learnable parameters for the quantum circuit (4 wires × 2 angles)
        self.params = nn.Parameter(torch.randn(4, 2))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Encode input: average pool and flatten to 16‑dim vector
        x_enc = self.encoder(x).view(bsz, 16)
        # Quantum forward pass
        out = quantum_circuit(x_enc, self.params)
        return self.norm(out)

__all__ = ["QFCModel"]
