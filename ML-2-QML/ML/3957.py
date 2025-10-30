"""Hybrid neural‑network combining a CNN encoder with a graph‑based quantum layer.

This module implements the `HybridNATGraphQNN` class, which merges a classical
convolutional encoder with a parameter‑shared quantum circuit.  The quantum
circuit is built with PennyLane and is driven by the output of the CNN.
Graph‑based entanglement is introduced via a simple linear‑chain adjacency
pattern, enabling the quantum layer to act as a learnable feature transformer.

Typical usage::

    import torch
    from quantum_nat_gen275 import HybridNATGraphQNN

    model = HybridNATGraphQNN()
    images = torch.randn(8, 1, 28, 28)
    logits = model(images)  # shape (8, 4)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pennylane as qml
from quantum_circuit import create_quantum_circuit

class HybridNATGraphQNN(nn.Module):
    """Hybrid CNN + graph‑based quantum layer.

    Parameters
    ----------
    n_wires : int
        Number of qubits in the quantum circuit (also the dimensionality of
        the feature vector after the CNN encoder).
    circuit_depth : int, optional
        Number of variational layers in the quantum circuit.  Currently
        implemented as a single rotation layer followed by entanglement.
    graph_threshold : float, optional
        Threshold for graph‑based adjacency; not used in the current
        implementation but kept for API compatibility.
    """

    def __init__(
        self,
        n_wires: int = 4,
        circuit_depth: int = 1,
        graph_threshold: float = 0.95,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.circuit_depth = circuit_depth
        self.graph_threshold = graph_threshold

        # Classical encoder (mirrors the original QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Project to the quantum feature dimensionality
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, n_wires),
        )
        self.norm = nn.BatchNorm1d(n_wires)

        # Quantum‑specific parameters
        # One rotation parameter per qubit; will be shared across all samples
        self.q_weights = nn.Parameter(torch.randn(n_wires))
        # Fixed linear‑chain adjacency (graph)
        self.adjacency = torch.eye(n_wires, dtype=torch.int64)
        for i in range(n_wires - 1):
            self.adjacency[i, i + 1] = 1
            self.adjacency[i + 1, i] = 1

        # Quantum circuit factory
        self.quantum_circuit = create_quantum_circuit(n_wires, self.adjacency)

        # Final linear classifier
        self.final_fc = nn.Linear(n_wires * 2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 4).
        """
        batch_size = x.shape[0]
        feats = self.features(x)
        flat = feats.view(batch_size, -1)
        qs = self.fc(flat)
        qs = self.norm(qs)  # (B, n_wires)

        # Apply quantum layer for each sample in the batch
        quantum_outs = torch.stack(
            [self.quantum_circuit(qs[i], self.q_weights) for i in range(batch_size)]
        )  # (B, n_wires)

        # Concatenate classical and quantum representations
        combined = torch.cat([qs, quantum_outs], dim=1)  # (B, 2*n_wires)

        logits = self.final_fc(combined)  # (B, 4)
        return logits

__all__ = ["HybridNATGraphQNN"]
