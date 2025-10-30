import pennylane as qml
import torch
import numpy as np

class HybridQuantumClassifier:
    """
    Pure quantum implementation of the hybrid classifier.
    Uses a variational two‑qubit circuit and PennyLane's automatic
    differentiation to compute probabilities from raw logits.
    """
    def __init__(self, n_qubits: int = 2, wires: int = 2, shots: int = 200):
        self.n_qubits = n_qubits
        self.wires = wires
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=wires, shots=shots)
        self.params = torch.nn.Parameter(torch.randn(n_qubits))
        self.qnode = qml.qnode(
            self.dev,
            interface="torch",
            diff_method="parameter-shift",
            batch_mode=True,
        )

    def circuit(self, x: torch.Tensor, params: torch.Tensor):
        """Variational circuit that maps input logits into an expectation."""
        for i in range(self.n_qubits):
            qml.RY(x[i] * params[i], wires=i)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that maps a batch of logits ``x`` (shape
        ``(batch, n_qubits)``) to a probability per sample.
        """
        probs = self.qnode(x, self.params)
        return torch.sigmoid(probs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

__all__ = ["HybridQuantumClassifier"]
