import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Classical QCNN-inspired feature extractor
class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# Hybrid quantum head
from quantum_module import QuantumHybridCircuit

class HybridQuantumLayer(nn.Module):
    """Differentiable quantum layer that forwards activations through a parameterised circuit."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.quantum_circuit = QuantumHybridCircuit(n_qubits, backend, shots, shift)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        thetas = inputs.detach().cpu().numpy().reshape(-1)
        expectations = self.quantum_circuit.run(thetas)
        return torch.tensor(expectations, dtype=torch.float32)

class HybridQCNNNet(nn.Module):
    """Hybrid network combining classical QCNN feature extraction with a QCNN‑inspired quantum head."""
    def __init__(self, n_qubits: int = 2, backend=None, shots: int = 100, shift: float = np.pi / 2) -> None:
        super().__init__()
        self.feature_extractor = QCNNModel()
        self.quantum_head = HybridQuantumLayer(n_qubits, backend, shots, shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(inputs)
        logits = self.quantum_head(features)
        probs = torch.sigmoid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["QCNNModel", "HybridQuantumLayer", "HybridQCNNNet"]
