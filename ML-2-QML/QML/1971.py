import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuantumNATHybridModel(nn.Module):
    """
    Quantum counterpart of the hybrid model: encodes pooled image features
    into qubits, runs a variational ansatz with a dense entanglement pattern,
    and measures the qubits to produce a 4‑dimensional feature vector.
    """
    def __init__(self, n_qubits=4, n_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.params = nn.Parameter(0.01 * torch.randn(n_layers, n_qubits, 3))
        self.encoder = nn.Linear(16, n_qubits)
        self.norm = nn.BatchNorm1d(n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        pooled = F.avg_pool2d(x, kernel_size=6, stride=6).view(bsz, -1)
        angles = self.encoder(pooled)
        outputs = []
        for i in range(bsz):
            outputs.append(self._circuit(angles[i]))
        out_tensor = torch.stack(outputs, dim=0)
        return self.norm(out_tensor)

    def _circuit(self, angles):
        @qml.qnode(self.dev, interface="torch")
        def circuit():
            for i in range(self.n_qubits):
                qml.RX(angles[i], wires=i)
            for l in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(self.params[l, i, 0], wires=i)
                    qml.RZ(self.params[l, i, 1], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit()

__all__ = ["QuantumNATHybridModel"]
