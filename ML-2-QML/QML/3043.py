# GraphQNN__gen316.py – quantum implementation

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import networkx as nx
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qutip as qt
import numpy as np

# --------------------------------------------------------------------------- #
# Helper utilities for random unitary generation and state manipulation
# --------------------------------------------------------------------------- #

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Return a random unitary on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary, _ = np.linalg.qr(matrix)
    qobj = qt.Qobj(unitary)
    qobj.dims = [[2] * num_qubits, [2] * num_qubits]
    return qobj


def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    """Generate synthetic training pairs for a target unitary."""
    dataset: list[tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        # random pure state
        state_vec = np.random.normal(size=(2 ** num_qubits,)) + 1j * np.random.normal(size=(2 ** num_qubits,))
        state_vec /= np.linalg.norm(state_vec)
        state = qt.Qobj(state_vec)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: list[int], samples: int):
    """Create a random quantum graph network with per‑layer unitaries."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[qt.Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), qt.qeye(2 ** (num_outputs - 1)))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    """Swap qubit registers in a multi‑qubit operator."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


# --------------------------------------------------------------------------- #
# Forward propagation helpers
# --------------------------------------------------------------------------- #

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    """Partial trace keeping the specified qubits."""
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    """Partial trace removing the specified qubits."""
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Apply a layer’s unitary and trace out the input qubits."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, qt.qeye(2))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[tuple[qt.Qobj, qt.Qobj]],
) -> list[list[qt.Qobj]]:
    """Propagate a batch of quantum states through the network."""
    stored_states: list[list[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# Quantum graph neural network module
# --------------------------------------------------------------------------- #

class GraphQNNHybrid(tq.QuantumModule):
    """
    Quantum graph neural network that mirrors the classical :class:`GraphQNNHybrid` API.

    Parameters
    ----------
    arch : Sequence[int]
        Layer sizes of the quantum GNN. Each layer corresponds to a block of
        parametrized gates acting on a subset of qubits.
    adjacency : Sequence[tuple[int, int]]
        Edges of the underlying graph. Each edge connects two qubit registers
        and is used to apply controlled gates during propagation.
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=20, wires=range(n_wires))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for w in range(self.n_wires):
                self.rx(qdev, wires=w)
                self.ry(qdev, wires=w)
                self.rz(qdev, wires=w)

    def __init__(self, arch: Sequence[int], adjacency: Sequence[tuple[int, int]]):
        super().__init__()
        self.arch = list(arch)
        self.adjacency = list(adjacency)

        # Encoder: a generic rotation that maps a classical node feature vector
        # to a product state over the qubits.
        self.encoder = tq.GeneralEncoder(
            tq.encoder_op_list_name_dict[f"{len(arch)}x{arch[0]}_ryzxy"]
        )

        # One QLayer per graph layer
        self.q_layers = nn.ModuleList([self.QLayer(n_wires=len(arch)) for _ in range(len(arch) - 1)])

        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(len(arch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Node feature tensor of shape (batch, n_nodes, node_feature_dim).

        Returns
        -------
        Tensor
            Measured qubit expectation values of shape (batch, n_nodes).
        """
        batch, n_nodes, _ = x.shape
        qdev = tq.QuantumDevice(n_wires=n_nodes, bsz=batch, device=x.device, record_op=True)

        # Encode raw node features into qubits
        self.encoder(qdev, x.view(batch, -1))

        # Propagate through each graph layer
        for layer in range(len(self.arch) - 1):
            self.q_layers[layer](qdev)
            # Apply controlled gates along the graph edges
            for i, j in self.adjacency:
                tqf.cx(qdev, wires=[i, j], static=self.static_mode, parent_graph=self.graph)

        out = self.measure(qdev)
        return self.norm(out)

    def compute_adjacency_from_states(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Convenience wrapper around :func:`fidelity_adjacency`."""
        return fidelity_adjacency(
            states,
            threshold,
            secondary=secondary,
            secondary_weight=secondary_weight,
        )

__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
