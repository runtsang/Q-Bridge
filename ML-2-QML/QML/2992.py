"""Hybrid graph‑quantum neural network implemented with torchquantum.

This quantum variant mirrors the classical HybridGraphQNN but operates on
quantum states.  It encodes input images with a parametric convolutional
encoder, propagates the encoded state through a graph‑structured
sequence of random unitary layers, and measures all qubits in the
Pauli‑Z basis.  The class shares the same public interface as its
classical counterpart, facilitating side‑by‑side comparison.

Key features
------------
* Convolutional quantum encoder (QFCModel style)
* Graph‑structured quantum layers with random unitaries
* Fidelity‑based adjacency graph construction
* Random network generator producing a list of unitary layers per layer
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import numpy as np
import networkx as nx
import scipy as sc
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qutip as qt

Tensor = torch.Tensor
Qobj = qt.Qobj


def _tensored_id(num_qubits: int) -> Qobj:
    """Identity operator with proper qudit dimensions."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity


def _tensored_zero(num_qubits: int) -> Qobj:
    """Zero projector for auxiliary qubits."""
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector


def _swap_registers(op: Qobj, source: int, target: int) -> Qobj:
    """Swap qubit registers in a tensor product operator."""
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> Qobj:
    """Generate a random unitary on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    matrix = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> Qobj:
    """Generate a random pure state on ``num_qubits`` qubits."""
    dim = 2 ** num_qubits
    amplitudes = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amplitudes /= sc.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: Qobj, samples: int) -> List[Tuple[Qobj, Qobj]]:
    """Create a dataset of input–target state pairs for a target unitary."""
    dataset: List[Tuple[Qobj, Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[Qobj]], List[Tuple[Qobj, Qobj]], Qobj]:
    """Generate a random quantum network and training data for its target unitary."""
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
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary


def _partial_trace_keep(state: Qobj, keep: Sequence[int]) -> Qobj:
    """Partial trace over all qubits except those in ``keep``."""
    if len(keep) == len(state.dims[0]):
        return state
    return state.ptrace(list(keep))


def _partial_trace_remove(state: Qobj, remove: Sequence[int]) -> Qobj:
    """Partial trace over qubits listed in ``remove``."""
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
    """Apply a single graph layer to ``input_state``."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[Qobj]],
    samples: Iterable[Tuple[Qobj, Qobj]],
) -> List[List[Qobj]]:
    """Forward‑propagate a batch of quantum states through the network."""
    stored_states: List[List[Qobj]] = []
    for sample, _ in samples:
        layerwise: List[Qobj] = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states


def state_fidelity(a: Qobj, b: Qobj) -> float:
    """Return the absolute squared overlap between pure states ``a`` and ``b``."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[Qobj],
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


class HybridGraphQNN(tq.QuantumModule):
    """Hybrid quantum graph‑neural network.

    Parameters
    ----------
    arch : Sequence[int]
        Size of each layer (number of qubits).  The first element
        determines the number of qubits used to encode the input.
    conv_cfg : dict | None
        Configuration for the convolutional encoder.  Defaults to the
        4‑qubit “4x4_ryzxy” encoder used in Quantum‑NAT.
    threshold : float
        Fidelity threshold for adjacency graph construction.
    secondary : float | None
        Optional secondary threshold; edges with fidelity between
        ``secondary`` and ``threshold`` receive ``secondary_weight``.
    secondary_weight : float
        Weight assigned to secondary edges.
    """

    def __init__(
        self,
        arch: Sequence[int],
        conv_cfg: dict | None = None,
        threshold: float = 0.8,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.arch = list(arch)
        self.threshold = threshold
        self.secondary = secondary
        self.secondary_weight = secondary_weight

        # Quantum convolutional encoder
        cfg = conv_cfg or {"name": "4x4_ryzxy"}
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[cfg["name"]])

        # Graph‑structured quantum layer
        self.q_layer = self.QLayer()

        # Measurement and normalisation
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.arch[-1])

    class QLayer(tq.QuantumModule):
        """Randomised quantum layer used in the graph."""

        def __init__(self) -> None:
            super().__init__()
            self.n_wires = 4
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

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: encode → graph layers → measurement → normalisation."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.arch[0], bsz=bsz, device=x.device, record_op=True)
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

    def random_network(self, samples: int) -> Tuple[List[int], List[List[Qobj]], List[Tuple[Qobj, Qobj]], Qobj]:
        """Generate a random network and training data for the target unitary."""
        return random_network(self.arch, samples)

    def feedforward(
        self,
        unitaries: Sequence[Sequence[Qobj]],
        samples: Iterable[Tuple[Qobj, Qobj]],
    ) -> List[List[Qobj]]:
        """Feedforward using externally supplied unitaries (for benchmarking)."""
        return feedforward(self.arch, unitaries, samples)

    def fidelity_adjacency(self, states: Sequence[Qobj]) -> nx.Graph:
        """Build adjacency graph from a list of quantum states."""
        return fidelity_adjacency(
            states,
            self.threshold,
            secondary=self.secondary,
            secondary_weight=self.secondary_weight,
        )

    def random_training_data(self, samples: int) -> List[Tuple[Qobj, Qobj]]:
        """Generate random training data for the target unitary."""
        _, _, dataset, _ = self.random_network(samples)
        return dataset


__all__ = ["HybridGraphQNN"]
