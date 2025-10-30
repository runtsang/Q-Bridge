import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedQuanvolution(nn.Module):
    """Variational quanvolutional filter using a shared RY‑rotation layer and a classical head."""

    def __init__(self):
        super().__init__()
        self.n_qubits = 4
        self.device = qml.device("default.qubit", wires=self.n_qubits)
        # QNode performing a shallow circuit with RY rotations and CNOT entanglement
        self.qnn = qml.QNode(self._circuit, self.device, interface="torch")
        self.classifier = nn.Linear(4 * 14 * 14, 10)

    def _circuit(self, patch):
        # patch: torch.Tensor of shape [4]
        for i in range(self.n_qubits):
            qml.RY(patch[i], wires=i)
        # entangling layer
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 28, 28]
        x = x.squeeze(1)  # [batch, 28, 28]
        batch_size = x.size(0)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r+2, c:c+2].view(batch_size, -1)  # [batch, 4]
                # Run each patch through the quantum circuit
                out = self.qnn(patch)  # [batch, 4]
                patches.append(out.unsqueeze(1))
        out = torch.cat(patches, dim=1)  # [batch, 14*14, 4]
        out = out.view(batch_size, -1)  # [batch, 4 * 14 * 14]
        logits = self.classifier(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["EnhancedQuanvolution"]
