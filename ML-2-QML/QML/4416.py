import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

# --------------------------------------------------------------------------- #
#  Pennylane quantum circuit with RX encoding and Ry‑CZ layers
# --------------------------------------------------------------------------- #
class QuantumCircuitQNode:
    """
    A QNode that implements the same ansatz as the Qiskit wrapper:
    RX encoding → repeated Ry + CZ layers.
    """
    def __init__(self, n_wires: int, depth: int, dev: qml.Device | None = None) -> None:
        self.n_wires = n_wires
        self.depth = depth
        self.dev = dev or qml.device("default.qubit", wires=n_wires)
        # Build the QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

    def _circuit(self, params: torch.Tensor) -> torch.Tensor:
        # params shape: (n_wires + depth * n_wires,)
        # RX encoding
        for i in range(self.n_wires):
            qml.RX(params[i], wires=i)
        idx = self.n_wires
        # Variational layers
        for _ in range(self.depth):
            for i in range(self.n_wires):
                qml.RY(params[idx], wires=i)
                idx += 1
            for i in range(self.n_wires - 1):
                qml.CZ(wires=[i, i + 1])
        # Expectation of Z on the first qubit
        return qml.expval(qml.PauliZ(0))

# --------------------------------------------------------------------------- #
#  Hybrid head – differentiable quantum expectation
# --------------------------------------------------------------------------- #
class HybridQuantumHead(nn.Module):
    def __init__(self, n_wires: int = 1, depth: int = 2) -> None:
        super().__init__()
        self.qnode = QuantumCircuitQNode(n_wires, depth).qnode

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        `inputs` should have shape (batch, n_wires + depth * n_wires).
        The QNode returns a scalar expectation per sample.
        """
        return self.qnode(inputs).unsqueeze(-1)

# --------------------------------------------------------------------------- #
#  Quantum‑only binary classifier
# --------------------------------------------------------------------------- #
class QuantumBinaryClassifier(nn.Module):
    """
    Mirrors the classical‑quantum hybrid structure but replaces the quantum
    expectation head with a Pennylane QNode.  The rest of the network
    (CNN + FC) is identical to the PyTorch implementation to enable
    direct comparison.
    """
    def __init__(self, n_wires: int = 1, depth: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid_head = HybridQuantumHead(n_wires, depth)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.drop2(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        prob = torch.sigmoid(self.hybrid_head(x))
        return torch.cat((prob, 1 - prob), dim=-1)

    @staticmethod
    def build_quantum_circuit(n_wires: int, depth: int):
        """
        Factory that returns the underlying QNode for use in other
        quantum‑only workflows, mirroring the `build_classifier_circuit`
        pattern from the reference pair.
        """
        return QuantumCircuitQNode(n_wires, depth).qnode

# --------------------------------------------------------------------------- #
#  Synthetic dataset generator (identical to the ML counterpart)
# --------------------------------------------------------------------------- #
def generate_binary_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a dataset where labels are determined by the sign of
    sin(sum(x)), creating a non‑linear decision boundary that is
    challenging for classical models and well‑suited for quantum
    expectation heads.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = (np.sin(angles) > 0).astype(np.float32)
    return x, y

__all__ = [
    "QuantumCircuitQNode",
    "HybridQuantumHead",
    "QuantumBinaryClassifier",
    "generate_binary_superposition_data",
]
