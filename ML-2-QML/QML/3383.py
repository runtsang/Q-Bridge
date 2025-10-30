"""Hybrid quantum graph neural network with quanvolution preprocessing.

This module mirrors the classical implementation but replaces
convolutional patches with a variational quantum kernel
and propagates quantum states through a layered unitary network.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
import torchquantum as tq


def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Absolute squared overlap between pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, a), (j, b) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(a, b)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> qt.Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Variational 2×2 patch encoder using a random two‑qubit layer."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class GraphQNNHybridQuantum(tq.QuantumModule):
    """Quantum graph neural network that embeds nodes via a quanvolution filter
    and propagates through a layered unitary network."""
    def __init__(self, qnn_arch: Sequence[int]) -> None:
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.qfilter = QuantumQuanvolutionFilter()
        self.unitaries: List[List[qt.Qobj]] = [[]]  # placeholder, filled in random_network

    def _layer_channel(
        self,
        layer: int,
        state: qt.Qobj,
    ) -> qt.Qobj:
        in_f, out_f = self.qnn_arch[layer - 1], self.qnn_arch[layer]
        # Prepare state with zero ancilla for new outputs
        state = qt.tensor(state, _tensored_zero(out_f))
        unitary = self.unitaries[layer][0].copy()
        for gate in self.unitaries[layer][1:]:
            unitary = gate * unitary
        new_state = unitary * state * unitary.dag()
        return _partial_trace_remove(new_state, range(in_f))

    def forward(self, x: torch.Tensor, adjacency: nx.Graph) -> qt.Qobj:
        """
        x: node feature tensor (batch, nodes, 4)
        adjacency: graph adjacency (ignored in this toy forward)
        """
        # Encode each node via quantum quanvolution
        batch, nodes, _ = x.shape
        patches = x.view(batch * nodes, 1, 28, 28)
        qfeat = self.qfilter(patches)  # (batch*nodes, 4)
        qfeat = qfeat.view(batch, nodes, -1)

        # Flatten to a single state per batch
        state = qt.tensor(_tensored_zero(self.qnn_arch[0]))
        for n in range(nodes):
            state = qt.tensor(state, qfeat[:, n, :])
        # Propagate through layers
        for layer in range(1, len(self.qnn_arch)):
            state = self._layer_channel(layer, state)
        return state

    @staticmethod
    def random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
        """Generate random unitaries and synthetic training data."""
        target_unitary = _random_qubit_unitary(qnn_arch[-1])
        training_data = random_training_data(target_unitary, samples)

        unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(qnn_arch)):
            num_inputs = qnn_arch[layer - 1]
            num_outputs = qnn_arch[layer]
            layer_ops: List[qt.Qobj] = []
            for output in range(num_outputs):
                op = _random_qubit_unitary(num_inputs + 1)
                if num_outputs > 1:
                    op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                    op = _swap_registers(op, num_inputs, num_inputs + output)
                layer_ops.append(op)
            unitaries.append(layer_ops)

        return list(qnn_arch), unitaries, training_data, target_unitary

    @staticmethod
    def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        """Generate synthetic training data for a target unitary."""
        dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
        num_qubits = len(unitary.dims[0])
        for _ in range(samples):
            state = _random_qubit_state(num_qubits)
            dataset.append((state, unitary * state))
        return dataset

__all__ = [
    "state_fidelity",
    "fidelity_adjacency",
    "QuantumQuanvolutionFilter",
    "GraphQNNHybridQuantum",
]
