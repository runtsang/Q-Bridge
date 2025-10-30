"""Quantum implementation of the hybrid graph‑quantum neural network.

The module retains the core QNN utilities from the original
GraphQNN.py while adding a quantum‑enhanced encoder inspired by
Quantum‑NAT.  The resulting :class:`GraphQNNHybrid`
is a :class:`torchquantum.QuantumModule` that maps 2‑D inputs to a set of
Pauli‑Z measurement outcomes.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Core QNN state propagation (adapted from the original QNN code)
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Return a tensor‑product identity on ``num_qubits``."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Projector onto the all‑zero state on ``num_qubits``."""
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    """Swap two qubits inside a tensor product operator."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    """Generate a Haar‑random unitary on ``num_qubits``."""
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Sample a random pure state on ``num_qubits``."""
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate training pairs (state, target_state) for a target unitary."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    """Create a random QNN and its training data."""
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

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    """Keep only the qubits listed in ``keep``."""
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    """Remove qubits indexed in ``remove``."""
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int, input_state: qt.Qobj) -> qt.Qobj:
    """Apply a single QNN layer to ``input_state``."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    """Run a forward pass through the QNN."""
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared fidelity between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    """Build a weighted graph from state fidelities."""
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
# Quantum‑enhanced hybrid model
# --------------------------------------------------------------------------- #

class GraphQNNHybrid(tq.QuantumModule):
    """
    Quantum hybrid model that mirrors the classical :class:`GraphQNNHybrid`
    but replaces the GNN with a variational circuit on a tensor‑product state.
    The image is first encoded into a set of qubits with a
    ``GeneralEncoder`` (Quantum‑NAT style), then processed by a
    ``QLayer`` that contains random gates and parameterised rotations.
    The measurement returns a vector of Pauli‑Z outcomes.
    """

    class QLayer(tq.QuantumModule):
        """Variational layer with random gates and trainable single‑qubit rotations."""

        def __init__(self, n_wires: int = 4):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.crx = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=3)
            self.crx(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, encoder_cfg: dict | None = None, layer_cfg: dict | None = None):
        super().__init__()
        # Default encoder from Quantum‑NAT
        encoder_cfg = encoder_cfg or {"encoder_op": "4x4_ryzxy"}
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[encoder_cfg["encoder_op"]])
        # Variational layer
        layer_cfg = layer_cfg or {"n_wires": 4}
        self.q_layer = self.QLayer(n_wires=layer_cfg["n_wires"])
        # Measurement and post‑processing
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(layer_cfg["n_wires"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the image, run the variational circuit, and return normalized Z‑counts."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.q_layer.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Pool the image to a feature vector matching the encoder's input size
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    # --------------------------------------------------------------------- #
    # Utility wrappers that expose the same interface as the original GraphQNN
    # --------------------------------------------------------------------- #
    def generate_random_network(self, samples: int = 10) -> tuple:
        """Return a random QNN architecture and training data."""
        arch, unitaries, training_data, target_unitary = random_network([4], samples)
        return arch, unitaries, training_data, target_unitary

    def compute_fidelity_graph(self, samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> nx.Graph:
        """Build a graph from the final state of the QNN."""
        arch, unitaries, training_data, target_unitary = random_network([4], samples)
        activations = feedforward(arch, unitaries, samples)
        final_states = [act[-1] for act in activations]
        return fidelity_adjacency(final_states, threshold=0.8)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNNHybrid",
]
