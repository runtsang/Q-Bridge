import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np
from typing import Iterable, List, Tuple


class QuantumClassifierModel:
    """
    Hybrid variational classifier implemented with Pennylane.
    Encodes data via RX gates, applies a depth‑controlled ansatz,
    measures Z on each qubit, and feeds results to a classical head.
    """

    def __init__(
        self,
        num_qubits: int,
        depth: int,
        device: str = "default.qubit",
        shots: int = 1024,
    ):
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = device
        self.shots = shots
        self.dev = qml.device(self.device, wires=self.num_qubits, shots=self.shots)

        # Parameters for the ansatz
        self.params = nn.Parameter(torch.randn(self.depth, self.num_qubits))
        # Classical post‑processing head
        self.head = nn.Linear(self.num_qubits, 2)

        self.encoding = list(range(self.num_qubits))
        self.weight_sizes = [
            self.params.numel() + self.head.weight.numel() + self.head.bias.numel()
        ]
        self.observables = [qml.PauliZ(i) for i in range(self.num_qubits)]

        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            # Data encoding
            for i in range(self.num_qubits):
                qml.RX(x[i], wires=i)

            # Variational ansatz
            for d in range(self.depth):
                for i in range(self.num_qubits):
                    qml.RY(params[d, i], wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])

            # Measurements
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        return circuit

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: quantum evaluation followed by a linear head.
        """
        X = X.to(torch.float32).unsqueeze(0)  # batch of one for simplicity
        quantum_out = self.circuit(X.squeeze(0), self.params)
        logits = self.head(quantum_out)
        return logits

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 10,
        lr: float = 0.01,
    ) -> None:
        """
        Train both quantum ansatz and classical head with Adam.
        """
        optimizer = torch.optim.Adam(
            list(self.params) + list(self.head.parameters()), lr=lr
        )
        criterion = nn.CrossEntropyLoss()
        self.circuit.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = self.forward(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Return class indices for the input batch.
        """
        self.circuit.eval()
        self.head.eval()
        with torch.no_grad():
            logits = self.forward(X)
            return torch.argmax(logits, dim=1).cpu()

    def get_metadata(self) -> Tuple[Iterable[int], List[int], List[int]]:
        """
        Return encoding indices, weight sizes and observables for consistency.
        """
        return self.encoding, self.weight_sizes, self.observables


__all__ = ["QuantumClassifierModel"]
