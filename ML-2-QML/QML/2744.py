"""Quantum module that implements a GraphQNN on top of a Quantum‑NAT style encoder.

The class `QuantumNATGraphQNN` inherits from `torchquantum.QuantumModule` and
provides:
  * A simple rotation‑based feature encoder that maps the first `n_qubits`
    components of a classical feature vector to single‑qubit RY gates.
  * A stack of `n_layers` random unitary layers (`torchquantum.RandomLayer`).
  * A target unitary that is applied to the all‑zeros state; intermediate
    fidelities against this target are recorded.
  * A measurement of all qubits in the Z basis and a batch‑wise batch norm.
  * A helper that builds a fidelity‑based adjacency graph over the batch of
    final states, which can be used for clustering or transfer learning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import networkx as nx
from typing import List, Tuple

class QuantumNATGraphQNN(tq.QuantumModule):
    """Quantum encoder + GraphQNN layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the device.
    n_layers : int
        Number of random layers to apply after encoding.
    seed : int | None
        Random seed for reproducibility.
    """

    def __init__(self, n_qubits: int = 4, n_layers: int = 3, seed: int | None = 42):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Random seed for reproducible unitaries
        if seed is not None:
            torch.manual_seed(seed)

        # Encoder: simple RY rotations per qubit using the first n_qubits features
        self.encoder = lambda qdev, x: self._encode_features(qdev, x)

        # Random unitary layers
        self.layers = nn.ModuleList(
            [tq.RandomLayer(n_ops=20, wires=list(range(n_qubits))) for _ in range(n_layers)]
        )

        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_qubits)

        # Generate a random target unitary and its state on |0...0>
        dim = 2 ** n_qubits
        random_matrix = torch.randn(dim, dim, dtype=torch.cfloat)
        target_unitary, _ = torch.linalg.qr(random_matrix)
        self.target_unitary = target_unitary
        zero_state = torch.zeros(dim, dtype=torch.cfloat)
        zero_state[0] = 1.0
        self.target_state = torch.matmul(self.target_unitary, zero_state)

    def _encode_features(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        """Encode the first `n_qubits` components of `x` into RY rotations."""
        for i in range(self.n_qubits):
            # Use the i-th feature as the rotation angle
            tq.RY(has_params=True, trainable=False)(qdev, wires=[i], params=x[:, i])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], nx.Graph]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Classical feature vector of shape (batch, feature_dim)
            where feature_dim >= n_qubits.

        Returns
        -------
        out : torch.Tensor
            Measurement outcome of shape (batch, n_qubits).
        fidelities : List[torch.Tensor]
            List of average fidelities per layer (length n_layers).
        graph : nx.Graph
            Fidelity‑based adjacency graph over the batch of final states.
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits, bsz=bsz, device=x.device, record_op=True
        )

        # Encode features
        self.encoder(qdev, x)

        fidelities: List[torch.Tensor] = []

        for layer in self.layers:
            layer(qdev)
            # Get the state vector for the batch
            state_vec = qdev.state_vector  # shape (bsz, 2**n_qubits)
            # Compute overlap with target state
            overlap = torch.sum(torch.conj(state_vec) * self.target_state, dim=1)
            fid = torch.abs(overlap) ** 2
            fidelities.append(fid.mean())

        # Final measurement
        out = self.measure(qdev)
        out = self.norm(out)

        # Build adjacency graph from the final state vectors
        final_state = qdev.state_vector  # shape (bsz, 2**n_qubits)
        graph = self._fidelity_adjacency(final_state, threshold=0.8)

        return out, fidelities, graph

    @staticmethod
    def _fidelity_adjacency(states: torch.Tensor, threshold: float) -> nx.Graph:
        """
        Build a weighted adjacency graph from state fidelities.

        Parameters
        ----------
        states : torch.Tensor
            Tensor of shape (batch, 2**n_qubits) containing state vectors.
        threshold : float
            Fidelity threshold for adding an edge with weight 1.

        Returns
        -------
        graph : nx.Graph
            Graph with nodes as batch indices and edges weighted by fidelity.
        """
        n = states.shape[0]
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                fid = torch.abs(torch.sum(torch.conj(states[i]) * states[j])) ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
        return graph
