import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
from typing import Tuple

class QLayerQuantum(nn.Module):
    """
    Quantum gate layer implemented with Pennylane.
    Produces a vector of length hidden_dim by measuring expectation values
    of PauliZ on each qubit and applying a linear transformation.
    """
    def __init__(self, n_qubits: int, hidden_dim: int, device: str = 'cpu'):
        super().__init__()
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        self.device = device

        # Classical linear mapping after measurement
        self.linear = nn.Linear(n_qubits, hidden_dim)

        # QNode
        self.qnode = qml.QNode(self._circuit,
                               device=qml.device('default.qubit', wires=n_qubits),
                               interface='torch')

    def _circuit(self, params: torch.Tensor) -> Tuple[float,...]:
        # params shape: (n_qubits,)
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i)
        # CNOT chain
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # Measure expectation of PauliZ on each wire
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, n_qubits)
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            out = self.qnode(x[i])
            out = self.linear(out)
            outputs.append(out)
        return torch.stack(outputs, dim=0)

__all__ = ["QLayerQuantum"]
