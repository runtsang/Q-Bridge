"""Hybrid quantum kernel module using TorchQuantum and Qiskit‑style self‑attention."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from typing import Sequence
import qutip as qt
import networkx as nx
import scipy as sc

class HybridKernel(tq.QuantumModule):
    """
    Quantum hybrid kernel.
    Encodes data with a parameterized rotation layer, applies a quantum
    self‑attention circuit, then evaluates the absolute amplitude overlap.
    The class also offers a graph construction based on state fidelity.
    """

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x and y on the device, then apply inverse to compute overlap."""
        q_device.reset_states(x.shape[0])
        # Encode x
        for i in range(self.n_wires):
            q_device.ry(x[:, i], wires=[i])
        # Self‑attention entanglement
        for i in range(self.n_wires - 1):
            q_device.crx(x[:, i], wires=[i, i + 1])
        # Encode y with negative parameters
        for i in range(self.n_wires):
            q_device.ry(-y[:, i], wires=[i])
        for i in range(self.n_wires - 1):
            q_device.crx(-y[:, i], wires=[i, i + 1])

    def forward_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return kernel value as absolute amplitude overlap."""
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.forward_kernel(x, y).item() for y in b] for x in a])

    def _random_qubit_state(self, num_qubits: int) -> qt.Qobj:
        dim = 2 ** num_qubits
        amps = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
        amps /= sc.linalg.norm(amps)
        state = qt.Qobj(amps)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        return state

    def fidelity_adjacency(self, states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for i, a in enumerate(states):
            for j in range(i + 1, len(states)):
                fid = abs((a.dag() * states[j])[0, 0]) ** 2
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph

__all__ = ["HybridKernel"]
