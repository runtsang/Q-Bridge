import pennylane as qml
import pennylane.numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure Pennylane uses PyTorch tensors for autograd
qml.torch.set_backend('torch')

def build_classifier_circuit(num_qubits: int, depth: int):
    """
    Construct a variational quantum circuit with explicit encoding and
    depth‑controlled ansatz.  Returns a QNode that can be called with a
    torch tensor of angles.
    """
    dev = qml.device('qiskit.aer', wires=num_qubits)

    @qml.qnode(dev, interface='torch')
    def circuit(x, weights):
        # Encoding
        for i in range(num_qubits):
            qml.RX(x[i], wires=i)

        # Variational layers
        idx = 0
        for _ in range(depth):
            for j in range(num_qubits):
                qml.RY(weights[idx], wires=j)
                idx += 1
            for j in range(num_qubits - 1):
                qml.CZ(wires=[j, j + 1])

        # Measure Z on first two qubits to obtain logits
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    return circuit


class QuantumHybridLayer(nn.Module):
    """
    Quantum head that maps a vector of classical angles to two logits via
    a variational circuit.  Parameters are trainable and gradients are
    obtained automatically by Pennylane.
    """
    def __init__(self, num_qubits: int, depth: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        # Trainable variational parameters
        self.weights = nn.Parameter(torch.randn(num_qubits * depth))
        # Build the circuit
        self.circuit = build_classifier_circuit(num_qubits, depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (batch, num_qubits) containing rotation angles.
        Returns logits of shape (batch, 2).
        """
        batch_size = x.shape[0]
        logits = torch.stack([self.circuit(sample, self.weights) for sample in x])
        return logits


class HybridQuantumBinaryClassifier(nn.Module):
    """
    Classical CNN followed by a quantum variational head.  The CNN backbone
    is identical to the classical version; the head is replaced by
    ``QuantumHybridLayer``.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)  # 4 features feed into the quantum head

        self.quantum_head = QuantumHybridLayer(num_qubits=4, depth=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        logits = self.quantum_head(x)
        probs = F.softmax(logits, dim=-1)
        return probs


__all__ = ["HybridQuantumBinaryClassifier", "QuantumHybridLayer", "build_classifier_circuit"]
