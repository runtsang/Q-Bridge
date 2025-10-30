"""Quantum module for HybridQuantumNATGraph.

This module contains the quantum component used by the hybrid model.  
It relies on torchquantum for device simulation and gate definition.

Classes:
    _QuantumLayer: Parameterized RX + CNOT entanglement based on adjacency.
    QuantumModule: Encodes classical activations into a 4‑qubit circuit and measures Pauli‑Z.
    _graph_for_fidelity: Utility that builds a graph from state fidelities.
"""

from __future__ import annotations

import networkx as nx
import torch
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Utility: Build adjacency graph from fidelity between activation vectors
# --------------------------------------------------------------------------- #
def _graph_for_fidelity(states: torch.Tensor, threshold: float = 0.95) -> nx.Graph:
    """Return a graph where edges connect states with fidelity ≥ threshold."""
    graph = nx.Graph()
    num_nodes = states.shape[1]
    graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            fid = torch.dot(states[:, i], states[:, j]) / (
                torch.norm(states[:, i]) + 1e-12
            ) / (torch.norm(states[:, j]) + 1e-12)
            if fid >= threshold:
                graph.add_edge(i, j)
    return graph

# --------------------------------------------------------------------------- #
# Quantum sub‑modules
# --------------------------------------------------------------------------- #
class _QuantumLayer(tq.QuantumModule):
    """Simple RX‑CNOT layer whose parameters are the input activations.

    Parameters are fed directly from the classical graph activations.
    """
    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.rx = tq.RX(has_params=True, trainable=True)
        self.cnot = tq.CNOT()

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice, angles: torch.Tensor, adjacency: nx.Graph) -> None:
        # Apply RX gates with angles derived from classical activations
        for i in range(self.n_qubits):
            self.rx(qdev, wires=i, params=angles[:, i])
        # Entangle qubits according to the adjacency graph
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                if adjacency.has_edge(i, j):
                    self.cnot(qdev, wires=[i, j])

class QuantumModule(tq.QuantumModule):
    """Quantum module that encodes classical activations and outputs a 4‑dimensional vector."""
    def __init__(self, n_qubits: int = 4) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        # Encoder that maps a classical vector into a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.var_layer = _QuantumLayer(n_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)

    @tq.static_support
    def forward(self, activations: torch.Tensor, adjacency: nx.Graph) -> torch.Tensor:
        bsz = activations.shape[0]
        qdev = tq.QuantumDevice(
            n_wires=self.n_qubits,
            bsz=bsz,
            device=activations.device,
            record_op=False,
        )
        # Encode the classical activations into the quantum state
        self.encoder(qdev, activations)
        # Apply the variational layer that uses the activations as parameters
        self.var_layer(qdev, activations, adjacency)
        # Measure in the Pauli‑Z basis and return the expectation values
        out = self.measure(qdev)
        return out

__all__ = ["QuantumModule", "_graph_for_fidelity"]
