"""Quantum hybrid graph neural network inspired by GraphQNN and Quantum‑NAT.

The class ``GraphQNNHybridQuantum`` implements a variational circuit that mirrors
the classical feed‑forward network.  Each layer consists of a random unitary
followed by a small trainable block (RX, RY, RZ, CRX) similar to the Quantum‑NAT
QLayer.  The encoder maps a 2‑D graph image into a quantum state using a
general encoder.  The final measurement yields a real vector that is
normalised by a classical batch‑norm layer.

The module relies on :mod:`torchquantum` for device creation, gate application and
automatic differentiation.  It also exposes the same fidelity‑based graph
utility from the original QML seed, enabling direct comparison with the
classical version.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import qutip as qt

Tensor = torch.Tensor

# ----- Quantum utilities (adapted from the original QML module) -----


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


def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)


def _random_qubit_unitary(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    matrix = torch.randn(dim, dim) + 1j * torch.randn(dim, dim)
    unitary = torch.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj


def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = torch.randn(dim, 1) + 1j * torch.randn(dim, 1)
    amplitudes /= torch.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int):
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
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state


def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
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
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(
        layer_unitary * state * layer_unitary.dag(), range(num_inputs)
    )


def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
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
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# ----- Quantum hybrid graph neural network class -----


class GraphQNNHybridQuantum(tq.QuantumModule):
    """
    Quantum hybrid graph neural network.

    Parameters
    ----------
    qnn_arch : Sequence[int]
        Layer sizes for the variational network.
    n_wires : int, optional
        Number of qubits used for the feature encoder (default 4).
    """

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self, qnn_arch: Sequence[int], n_wires: int = 4):
        super().__init__()
        self.qnn_arch = list(qnn_arch)
        self.n_wires = n_wires
        # Encoder that maps a 2‑D image into a quantum state
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        # Build a random variational circuit mirroring the classical layers
        _, self.unitaries, _, self.target = random_network(qnn_arch, samples=10)
        self.q_layer = self.QLayer(n_wires=self.n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, 1, 28, 28) representing graph adjacency
            matrices rendered as single‑channel images.

        Returns
        -------
        torch.Tensor
            Normalised output of size (N, n_wires).
        """
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        # Feature encoding
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        # Variational layer
        self.q_layer(qdev)
        # Measurement
        out = self.measure(qdev)
        return self.norm(out)

    @staticmethod
    def build_random_graph(
        qnn_arch: Sequence[int], samples: int, threshold: float = 0.8
    ) -> nx.Graph:
        _, unitaries, _, target = random_network(qnn_arch, samples)
        states = [u[0] for u in unitaries[1]]  # first unitary per layer
        return fidelity_adjacency(states, threshold)

    @staticmethod
    def random_training_set(
        qnn_arch: Sequence[int], samples: int
    ) -> List[Tuple[qt.Qobj, qt.Qobj]]:
        _, _, dataset, _ = random_network(qnn_arch, samples)
        return dataset

    def train_on_dataset(
        self,
        dataset: List[Tuple[qt.Qobj, qt.Qobj]],
        lr: float = 1e-3,
        epochs: int = 10,
    ) -> None:
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            for state, target in dataset:
                optimizer.zero_grad()
                # Prepare device
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=1, device=target.device)
                # Encode the input state
                qdev.set_state(state)
                # Forward through the circuit
                self.q_layer(qdev)
                out = self.measure(qdev)
                loss = loss_fn(out, target)
                loss.backward()
                optimizer.step()
