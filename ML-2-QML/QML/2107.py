import torch
import pennylane as qml
from torch.nn import BatchNorm1d

class QuantumNATGen321:
    """Quantum model using a Pennylane variational circuit."""
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Trainable variational parameters
        self.params = torch.nn.Parameter(torch.randn(n_layers, n_qubits, 3))
        # QNode with torch interface
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")
        self.norm = BatchNorm1d(n_qubits)

    def _circuit(self, x: torch.Tensor, params: torch.Tensor):
        # Classical encoding: RX gates with input angles
        x_flat = x.view(-1)
        for j in range(self.n_qubits):
            angle = x_flat[j] if j < x_flat.shape[0] else 0.0
            qml.RX(angle, wires=j)
        # Variational layers
        for i in range(self.n_layers):
            for j in range(self.n_qubits):
                qml.RY(params[i, j, 0], wires=j)
                qml.RZ(params[i, j, 1], wires=j)
                qml.RX(params[i, j, 2], wires=j)
                if j < self.n_qubits - 1:
                    qml.CNOT(wires=[j, j + 1])
        # Measurement in the Z basis for all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, H, W). Flatten per sample.
        batch = x.shape[0]
        outputs = []
        for i in range(batch):
            sample = x[i].view(-1).to(x.device)
            out = self.qnode(sample, self.params)
            outputs.append(out)
        out_tensor = torch.stack(outputs)
        return self.norm(out_tensor)

__all__ = ["QuantumNATGen321"]
