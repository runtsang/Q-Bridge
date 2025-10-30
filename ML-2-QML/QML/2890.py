"""Quantum‑inspired graph neural network implemented with torchquantum.

The model mirrors the classical GraphQNNHybrid but replaces the
fully‑connected backbone with a parameterised quantum layer.  It
encodes node features into qubits via a general encoder, applies a
RandomLayer followed by a small set of trainable gates, and measures
all qubits.  Fidelity‑based adjacency utilities are provided for
state‑based graph construction."""
from __future__ import annotations

import itertools
from typing import List, Tuple, Sequence, Iterable

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qutip as qt
import networkx as nx
import numpy as np

Tensor = torch.Tensor
Qobj = qt.Qobj


# --------------------------------------------------------------------------- #
#   Utility functions (random unitary, training data, feed‑forward)
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> Qobj:
    """Return a random unitary as a qutip Qobj."""
    dim = 2 ** num_qubits
    matrix = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    unitary = qt.Qobj(np.linalg.qr(matrix)[0])
    return unitary


def random_training_data(unitary: Qobj, samples: int) -> List[Tuple[Qobj, Qobj]]:
    """Generate random input states and their unitary‑transformed targets."""
    dataset: List[Tuple[Qobj, Qobj]] = []
    num_qubits = unitary.dims[0][0]
    for _ in range(samples):
        state = qt.rand_ket(num_qubits)
        target = unitary * state
        dataset.append((state, target))
    return dataset


def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random quantum network and a training set."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), qt.qeye(2 ** (num_outputs - 1)))
                op = qt.tensor(op, qt.swap(num_inputs, num_inputs + output))
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary


def _partial_trace_keep(state: Qobj, keep: Sequence[int]) -> Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: Qobj, remove: Sequence[int]) -> Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)


def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Qobj]],
    layer: int,
    input_state: Qobj,
) -> Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, qt.qeye(2 ** num_outputs))
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Qobj]],
    samples: Iterable[Tuple[Qobj, Qobj]],
) -> List[List[Qobj]]:
    """Propagate a batch of input states through the quantum network."""
    stored_states: List[List[Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: Qobj, b: Qobj) -> float:
    """Return the absolute squared overlap of two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Create a weighted graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
#   Quantum GraphQNNHybrid model
# --------------------------------------------------------------------------- #
class QGraphQNNHybrid(tq.QuantumModule):
    """Quantum graph neural network with a trainable RandomLayer and
    a small set of parameterised gates.  The model encodes node
    features into qubits, applies the quantum layer, and measures all
    qubits."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice) -> None:
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, n_nodes: int):
        super().__init__()
        self.n_wires = n_nodes
        # Use a general encoder that accepts the same number of wires as nodes
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"][:n_nodes])
        self.q_layer = self.QLayer(n_nodes)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Node feature matrix of shape (batch, nodes, in_features).

        Returns
        -------
        torch.Tensor
            Normalised measurement vector of shape (batch, n_wires).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Flatten node features into a single vector per sample
        pooled = x.mean(dim=1).view(bsz, -1)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = [
    "QGraphQNNHybrid",
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
]
