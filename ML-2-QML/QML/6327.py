import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Iterable, List

__all__ = ["HybridNATModel"]


class HybridNATModel(nn.Module):
    """
    Quantum implementation of the HybridNAT model using PennyLane.
    The network encodes classical features into qubits, applies a
    depth‑controlled variational ansatz, and measures Pauli‑Z
    expectation values.  A lightweight classical head maps the
    measurement outcomes to logits.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        depth: int = 3,
        num_classes: int = 4,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth
        self.num_classes = num_classes
        self.dev = qml.device(device, wires=n_qubits)
        # Learnable variational parameters
        self.params = nn.Parameter(torch.randn((depth, n_qubits)))
        self._build_circuit()
        # Classical head
        self.classifier = nn.Linear(n_qubits, num_classes)

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor):
            # Feature map: RX encoding
            for i in range(self.n_qubits):
                qml.RX(inputs[i], wires=i)
            # Variational layers
            idx = 0
            for _ in range(self.depth):
                for i in range(self.n_qubits):
                    qml.RY(params[idx, i], wires=i)
                    idx += 1
                for i in range(self.n_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            # Measurement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: x must be a batch of flattened feature vectors
        of shape (batch, n_qubits).  The circuit returns expectation
        values which are fed into a classical linear layer.
        """
        bsz = x.shape[0]
        out = self.circuit(x, self.params)  # shape (batch, n_qubits)
        out = torch.stack(out, dim=1)  # ensure correct shape
        logits = self.classifier(out)
        return logits

    @staticmethod
    def build_classifier_circuit(
        num_qubits: int, depth: int
    ) -> Tuple[qml.QNode, Iterable, Iterable, List[qml.PauliZ]]:
        """
        Recreates the incremental data‑uploading classifier using
        PennyLane.  Returns the circuit, encoding parameters, variational
        parameters, and measurement observables.
        """
        # Encoding parameters (classical placeholder)
        encoding = np.random.uniform(0, 2 * np.pi, size=num_qubits)
        # Variational parameters placeholder
        weights = np.zeros((depth, num_qubits))
        dev = qml.device("default.qubit", wires=num_qubits)

        @qml.qnode(dev, interface="numpy")
        def circuit():
            for i in range(num_qubits):
                qml.RX(encoding[i], wires=i)
            idx = 0
            for _ in range(depth):
                for i in range(num_qubits):
                    qml.RY(weights[idx, i], wires=i)
                    idx += 1
                for i in range(num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        observables = [qml.PauliZ(i) for i in range(num_qubits)]
        return circuit, encoding, weights, observables
