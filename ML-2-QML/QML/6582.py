"""Quantum transformer classifier using Pennylane.

This module implements a quantum‑centric variant of the `HybridQuantumClassifier`
originally defined in the classical module.  The circuit encodes each feature
vector into qubits, applies a variational layer with trainable Ry gates,
entangles the qubits with CNOTs, and measures the expectation of Z on each
wire.  The resulting tensor is passed through a classical linear head.
"""

import pennylane as qml
import torch
import torch.nn as nn
from typing import Iterable, List, Tuple

class HybridQuantumClassifier(nn.Module):
    """
    Quantum transformer classifier implemented with Pennylane.

    Parameters
    ----------
    num_features : int
        Dimensionality of the input token vector (must match the number of qubits).
    depth : int
        Number of variational layers in the circuit.
    n_wires : int
        Number of qubits used in the circuit.
    """

    def __init__(self, num_features: int, depth: int, n_wires: int):
        super().__init__()
        self.num_features = num_features
        self.depth = depth
        self.n_wires = n_wires

        # Device for simulation
        self.dev = qml.device("default.qubit", wires=self.n_wires)

        # Variational parameters
        self.params = nn.Parameter(torch.randn(self.depth, self.n_wires))

        # Classical head
        self.head = nn.Linear(self.n_wires, 2)

    def circuit(self, x, params):
        """Variational circuit for a single token."""
        for i in range(self.n_wires):
            qml.RX(x[i], wires=i)          # data encoding
        for layer in range(self.depth):
            for w in range(self.n_wires):
                qml.RY(params[layer, w], wires=w)
            for w in range(self.n_wires - 1):
                qml.CNOT(wires=[w, w + 1])
        return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, seq_len, num_features).
        """
        batch, seq_len, _ = x.shape
        outputs = []
        for i in range(seq_len):
            token = x[:, i, :]
            qnode = qml.QNode(
                lambda token: self.circuit(token, self.params),
                self.dev,
                interface="torch",
            )
            out = qnode(token)
            outputs.append(out)
        out = torch.stack(outputs, dim=1)  # (batch, seq_len, n_wires)
        out = out.mean(dim=1)  # pool over seq_len
        return self.head(out)

def build_classifier_circuit(
    num_features: int,
    depth: int,
    n_wires: int,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """
    Construct a quantum classifier and return metadata.

    Parameters
    ----------
    num_features : int
        Number of input features (must equal n_wires).
    depth : int
        Number of variational layers.
    n_wires : int
        Number of qubits.

    Returns
    -------
    nn.Module
        The quantum classifier instance.
    Iterable[int]
        Encoding indices (identity mapping).
    Iterable[int]
        Per‑layer parameter counts.
    List[int]
        Observable indices (class labels).
    """
    classifier = HybridQuantumClassifier(num_features, depth, n_wires)
    encoding = list(range(num_features))
    weight_sizes = [p.numel() for p in classifier.parameters() if p.requires_grad]
    observables = [0, 1]
    return classifier, encoding, weight_sizes, observables

__all__ = ["HybridQuantumClassifier", "build_classifier_circuit"]
