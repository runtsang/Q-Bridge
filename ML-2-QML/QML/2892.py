"""Quantum Graph Neural Network mirroring the classical API.

The quantum implementation uses torchquantum to encode each node’s
features into a shared quantum device, applies a variational layer,
and measures all qubits.  The design follows the structure of the
original QFCModel while integrating graph‑based operations.
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

Tensor = torch.Tensor


def _rand_unitary(num_qubits: int) -> tq.Qobj:
    """Return a random Haar‑distributed unitary on `num_qubits`."""
    dim = 2 ** num_qubits
    matrix = torch.randn(dim, dim, dtype=torch.complex64) + 1j * torch.randn(dim, dim, dtype=torch.complex64)
    q, _ = torch.linalg.qr(matrix)
    return tq.Qobj(q)


def random_training_data(unitary: tq.Qobj, samples: int) -> List[Tuple[tq.Qobj, tq.Qobj]]:
    """Generate synthetic state pairs (|ψ⟩, U|ψ⟩)."""
    data: List[Tuple[tq.Qobj, tq.Qobj]] = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        amp = torch.randn(2 ** num_qubits, dtype=torch.complex64)
        amp = amp / amp.norm()
        state = tq.Qobj(amp)
        state.dims = [[2] * num_qubits, [1] * num_qubits]
        data.append((state, unitary * state))
    return data


def random_network(arch: Sequence[int], samples: int):
    """Create random unitary layers for each graph depth."""
    target = _rand_unitary(arch[-1])
    training = random_training_data(target, samples)

    layers: List[List[tq.Qobj]] = [[]]
    for layer in range(1, len(arch)):
        in_f = arch[layer - 1]
        out_f = arch[layer]
        ops: List[tq.Qobj] = []
        for _ in range(out_f):
            ops.append(_rand_unitary(in_f + 1))
        layers.append(ops)

    return list(arch), layers, training, target


def state_fidelity(a: tq.Qobj, b: tq.Qobj) -> float:
    """Squared overlap of pure states a and b."""
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(
    states: Sequence[tq.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Construct a graph from quantum state fidelities."""
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(s_i, s_j)
        if fid >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


class GraphQNNGen156(tq.QuantumModule):
    """Quantum GNN that encodes node features and applies a variational layer."""

    class QLayer(tq.QuantumModule):
        """Variational block with parameterised single‑qubit rotations and a CNOT."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.rand = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)
            self.rz = tq.RZ(has_params=True, trainable=True)
            self.cnot = tq.CNOT()

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.rand(qdev)
            self.rx(qdev, wires=0)
            self.ry(qdev, wires=1)
            self.rz(qdev, wires=2)
            self.cnot(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)

    def __init__(self, arch: Sequence[int]):
        super().__init__()
        self.arch = list(arch)
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer(self.arch[-1])
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.arch[-1])

    def forward(self, x: Tensor) -> Tensor:
        """Encode input, run quantum layer, and return normalized measurements."""
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.arch[-1], bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, self.arch[-1])
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


__all__ = [
    "GraphQNNGen156",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "fidelity_adjacency",
]
