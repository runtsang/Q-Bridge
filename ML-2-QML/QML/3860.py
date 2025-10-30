"""GraphQNNHybrid: quantum‑state‑based graph neural network.

This module mirrors the classical implementation but replaces linear
layers with parameterised unitary blocks that act on quantum registers.
The interface stays identical, enabling a zero‑copy switch between
classical and quantum back‑ends for downstream experiments.
"""

from __future__ import annotations

import itertools
import numpy as np
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import qutip as qt

Qobj = qt.Qobj


def _tensored_id(num_qubits: int) -> Qobj:
    I = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    I.dims = [dims.copy(), dims.copy()]
    return I


def _tensored_zero(num_qubits: int) -> Qobj:
    zero = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    zero.dims = [dims.copy(), dims.copy()]
    return zero


def _swap_registers(op: Qobj, src: int, tgt: int) -> Qobj:
    if src == tgt:
        return op
    order = list(range(len(op.dims[0])))
    order[src], order[tgt] = order[tgt], order[src]
    return op.permute(order)


def _random_unitary(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    q, _ = np.linalg.qr(mat)
    return qt.Qobj(q)


def _random_state(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return qt.Qobj(vec, dims=[[2] * num_qubits, [1] * num_qubits])


def random_training_data(target: Qobj, samples: int) -> List[Tuple[Qobj, Qobj]]:
    data: List[Tuple[Qobj, Qobj]] = []
    for _ in range(samples):
        state = _random_state(len(target.dims[0]))
        data.append((state, target * state))
    return data


def random_network(arch: Sequence[int], samples: int):
    """Create a random network of unitary blocks and a training set."""
    target = _random_unitary(arch[-1])
    training = random_training_data(target, samples)

    unitaries: List[List[Qobj]] = [[]]
    for layer in range(1, len(arch)):
        nin, nout = arch[layer - 1], arch[layer]
        layer_ops: List[Qobj] = []
        for out in range(nout):
            op = _random_unitary(nin + 1)
            if nout > 1:
                op = qt.tensor(_random_unitary(nin + 1), _tensored_id(nout - 1))
                op = _swap_registers(op, nin, nin + out)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(arch), unitaries, training, target


def _partial_trace(state: Qobj, keep: Sequence[int]) -> Qobj:
    return state.ptrace(list(keep))


def _layer_channel(arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]],
                   layer: int, inp: Qobj) -> Qobj:
    nin, nout = arch[layer - 1], arch[layer]
    state = qt.tensor(inp, _tensored_zero(nout))
    gate = unitaries[layer][0]
    for g in unitaries[layer][1:]:
        gate = g * gate
    return _partial_trace(gate * state * gate.dag(), range(nin))


def feedforward(arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]],
                samples: Iterable[Tuple[Qobj, Qobj]]) -> List[List[Qobj]]:
    outputs: List[List[Qobj]] = []
    for inp, _ in samples:
        layer_states = [inp]
        cur = inp
        for layer in range(1, len(arch)):
            cur = _layer_channel(arch, unitaries, layer, cur)
            layer_states.append(cur)
        outputs.append(layer_states)
    return outputs


def state_fidelity(a: Qobj, b: Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2


def fidelity_adjacency(states: Sequence[Qobj], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(len(states)))
    for (i, sa), (j, sb) in itertools.combinations(enumerate(states), 2):
        f = state_fidelity(sa, sb)
        if f >= threshold:
            G.add_edge(i, j, weight=1.0)
        elif secondary is not None and f >= secondary:
            G.add_edge(i, j, weight=secondary_weight)
    return G


class GraphQNNHybrid:
    """Quantum‑state based hybrid network with a classical‑looking API."""
    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)
        self.unitaries: List[List[Qobj]] | None = None
        self.training: List[Tuple[Qobj, Qobj]] | None = None

    def initialize(self, samples: int = 100) -> None:
        _, self.unitaries, self.training, _ = random_network(self.arch, samples)

    def forward(self, state: Qobj) -> Qobj:
        cur = state
        for layer in range(1, len(self.arch)):
            cur = _layer_channel(self.arch, self.unitaries, layer, cur)
        return cur

    def fidelity_graph(self, threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
        activations = feedforward(self.arch, self.unitaries, self.training)
        states = [act[-1] for act in activations]
        return fidelity_adjacency(states, threshold,
                                   secondary=secondary,
                                   secondary_weight=secondary_weight)


__all__ = [
    "GraphQNNHybrid",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
