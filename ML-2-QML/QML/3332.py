"""Quantum‑only implementation of the hybrid model using Qiskit."""

from __future__ import annotations

import itertools
import numpy as np
import networkx as nx
import torch
import torch.nn as nn

from qiskit import QuantumCircuit, Aer, execute

class QuanvolutionHybridGraphQL(nn.Module):
    """
    Quantum‑only version of the hybrid model using Qiskit.
    """
    def __init__(self, num_classes: int = 10, threshold: float = 0.8,
                 secondary: float | None = None, hidden_dim: int = 32):
        super().__init__()
        self.num_classes = num_classes
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = 0.5
        self.hidden_dim = hidden_dim
        self.classifier = nn.Linear(4, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)
        self.backend = Aer.get_backend('statevector_simulator')

    def _state_fidelity(self, a: np.ndarray, b: np.ndarray) -> float:
        an = a / (np.linalg.norm(a) + 1e-12)
        bn = b / (np.linalg.norm(b) + 1e-12)
        return float(np.vdot(an, bn)**2)

    def _quantum_patch(self, data: np.ndarray) -> np.ndarray:
        """
        Apply a simple 4‑qubit variational circuit to a 2×2 patch.
        data: shape (4,) float values in [0, 1].
        Returns: expectation values of PauliZ on each qubit.
        """
        qc = QuantumCircuit(4)
        # Encode pixel intensities as Ry rotations
        for i, val in enumerate(data):
            qc.ry(val, i)
        # Simple variational layer
        for i in range(4):
            qc.ry(0.5, i)
        for i in range(3):
            qc.cx(i, i+1)
        # Measure all qubits
        qc.measure_all()
        job = execute(qc, self.backend, shots=1024)
        result = job.result()
        counts = result.get_counts()
        exp = np.zeros(4)
        for bitstring, freq in counts.items():
            bits = [int(b) for b in bitstring[::-1]]  # reverse order
            for i, bit in enumerate(bits):
                exp[i] += (1 - 2 * bit) * freq
        exp /= 1024
        return exp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        # Flatten patches into (B, 196, 4)
        patches = x.view(B, 1, H, W).permute(0, 2, 3, 1).reshape(B, 196, 4)
        quantum_features = []
        for b in range(B):
            vecs = []
            for patch in patches[b]:
                vec = self._quantum_patch(patch.detach().numpy())
                vecs.append(vec)
            quantum_features.append(np.stack(vecs, axis=0))  # (196,4)
        quantum_features = np.stack(quantum_features, axis=0)  # (B,196,4)
        logits_list = []
        for b in range(B):
            vecs = quantum_features[b]
            G = nx.Graph()
            G.add_nodes_from(range(196))
            for u, v in itertools.combinations(range(196), 2):
                fid = self._state_fidelity(vecs[u], vecs[v])
                if fid >= self.threshold:
                    G.add_edge(u, v, weight=1.0)
                elif self.secondary and fid >= self.secondary:
                    G.add_edge(u, v, weight=self.secondary_weight)
            node_feats = torch.tensor([vecs[n] for n in G.nodes], dtype=torch.float32)
            pooled = torch.mean(node_feats, dim=0)
            logits = self.classifier(pooled)
            logits_list.append(logits)
        logits = torch.stack(logits_list)
        return self.head(logits)

__all__ = ["QuanvolutionHybridGraphQL"]
