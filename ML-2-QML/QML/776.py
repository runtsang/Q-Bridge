import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridBinaryClassifier(nn.Module):
    """Hybrid CNN with a variational quantum circuit head for binary classification.

    The quantum head uses a parameterised two‑qubit circuit and returns
    the expectation value of Pauli‑Z on the first qubit. The result is
    passed through a sigmoid to obtain a probability.
    """
    def __init__(self, device: str = "default.qubit", shots: int = 1024):
        super().__init__()
        self.device = device
        self.shots = shots

        # Classical CNN backbone identical to the ML version
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )
        self.flatten = nn.Flatten()

        # Prepare a variational circuit on 2 qubits
        self.qnode = qml.QNode(self._quantum_circuit, device=self.device,
                               interface="torch", shots=self.shots)

        # Classical dense layer to map CNN output to a single parameter for the circuit
        self.param_layer = nn.Linear(64 * 8 * 8, 1)

    def _quantum_circuit(self, theta):
        """Two‑qubit parameterised circuit returning the expectation of Z on qubit 0."""
        qml.Hadamard(0)
        qml.Hadamard(1)
        qml.RY(theta, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        theta = self.param_layer(x)
        expval = self.qnode(theta)
        probs = torch.sigmoid(expval)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["HybridBinaryClassifier"]
