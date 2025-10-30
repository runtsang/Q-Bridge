import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class HybridQuantumClassifier(nn.Module):
    """Hybrid quantum–classical binary classifier.

    The convolutional backbone is identical to the classical version.
    The final linear head is replaced by a two‑qubit variational circuit
    whose expectation value of Z is used as a probability. A tiny
    trainable linear layer fine‑tunes the quantum output.
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
        self.fc3 = nn.Linear(84, 1)

        # Pennylane device and quantum node
        self.dev = qml.device("default.qubit", wires=2, shots=200)
        self.qnode = qml.QNode(HybridQuantumClassifier._variational_circuit,
                               device=self.dev,
                               interface="torch")

        # Small trainable head after the quantum expectation
        self.classifier = nn.Linear(1, 2)

    @staticmethod
    def _variational_circuit(theta):
        """Two‑qubit circuit parameterised by a single angle."""
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Barrier(wires=[0, 1])
        qml.RY(theta, wires=0)
        qml.RY(theta, wires=1)
        return qml.expval(qml.PauliZ(0))

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
        # Evaluate the quantum circuit for each sample
        qs = torch.stack([self.qnode(t) for t in x.squeeze()]).unsqueeze(-1)
        logits = self.classifier(qs)
        probs = F.softmax(logits, dim=-1)
        return probs
