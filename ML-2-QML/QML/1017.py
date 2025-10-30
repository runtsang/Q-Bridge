import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QFCModel(nn.Module):
    """Quantum variational model with a feature‑map encoder and 8‑qubit ansatz."""

    def __init__(self, n_qubits: int = 8, n_layers: int = 3) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.qlayer = self._build_qnode()
        self.classifier = nn.Linear(n_qubits, 4)
        self.norm = nn.BatchNorm1d(4)

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            # Feature map: encode first 4 features into first 4 qubits
            for i in range(4):
                qml.RY(x[i], wires=i)
            # Variational ansatz
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RY(params[layer, qubit], wires=qubit)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        # Global average pool to 4 features
        pooled = F.avg_pool2d(x, kernel_size=6).view(bsz, 4)
        # Initialize variational parameters
        params = torch.randn(self.n_layers, self.n_qubits, device=x.device, requires_grad=True)
        # Evaluate quantum circuit for each sample
        q_out = torch.stack([self.qlayer(pooled[i], params) for i in range(bsz)])
        q_out = q_out.reshape(bsz, self.n_qubits)
        out = self.classifier(q_out)
        return self.norm(out)

__all__ = ["QFCModel"]
