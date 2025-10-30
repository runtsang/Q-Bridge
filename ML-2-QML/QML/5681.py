import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Quantum quanvolution filter implemented with Pennylane."""
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x: torch.Tensor) -> torch.Tensor:
            qml.templates.AngleEmbedding(x, wires=range(n_qubits))
            for _ in range(2):
                qml.templates.StronglyEntanglingLayers(
                    torch.randn(2, n_qubits), wires=range(n_qubits)
                )
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r+2, c:c+2]
                patch_flat = patch.reshape(bsz, 4)
                feat = self.circuit(patch_flat)          # list of 4 tensors (batch,)
                feat = torch.stack(feat, dim=1)           # (batch, 4)
                patches.append(feat)
        return torch.cat(patches, dim=1)

class QuantumFullyConnectedLayer(nn.Module):
    """Parameterized quantum circuit acting as a fully‑connected layer."""
    def __init__(self, n_qubits: int = 1, n_layers: int = 3) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def circuit(params: torch.Tensor) -> torch.Tensor:
            qml.templates.AngleEmbedding(params, wires=range(n_qubits))
            for _ in range(n_layers):
                qml.templates.StronglyEntanglingLayers(
                    torch.randn(n_layers, n_qubits), wires=range(n_qubits)
                )
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_qubits)
        return self.circuit(x)

class QuanvolutionHybrid(nn.Module):
    """Hybrid quantum‑classical model: quantum quanvolution filter + quantum fully‑connected layer."""
    def __init__(self, n_fc_qubits: int = 1, n_fc_layers: int = 3) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qfc = QuantumFullyConnectedLayer(n_qubits=n_fc_qubits, n_layers=n_fc_layers)
        self.out = nn.Linear(1, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)          # (batch, 4*14*14)
        params = features[:, :self.qfc.n_qubits]  # first n_qubits as parameters
        q_out = self.qfc(params)            # (batch,)
        logits = self.out(q_out.unsqueeze(1))
        return F.log_softmax(logits, dim=-1)

class QuanvolutionClassifier(nn.Module):
    """Quantum classifier using a quantum filter and a classical linear head."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier", "QuanvolutionHybrid"]
