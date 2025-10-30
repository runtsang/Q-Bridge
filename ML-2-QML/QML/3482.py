"""Quantum Graph‑Quanvolution network.

This module implements a quantum‑inspired patch encoder followed by a
variational graph network.  It follows the same public API as the
classical version but uses torchquantum and qutip primitives.
"""

from __future__ import annotations

import itertools
from typing import Iterable, List, Sequence

import networkx as nx
import qutip as qt
import scipy as sc
import torch
import torch.nn as nn
import torchquantum as tq

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Quantum utilities
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    zero = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    zero.dims = [dims.copy(), dims.copy()]
    return zero

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

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Squared overlap of two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Build a weighted graph from fidelity between quantum states."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, si), (j, sj) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(si, sj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(len(state.dims[0])))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return state.ptrace(keep)

def _layer_channel(
    arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Apply a layer of unitary gates and trace out input qubits."""
    num_inputs = arch[layer - 1]
    num_outputs = arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(
    arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[qt.Qobj],
) -> List[List[qt.Qobj]]:
    """Propagate quantum states through the graph network."""
    outputs = []
    for sample in samples:
        layerwise = [sample]
        current = sample
        for layer in range(1, len(arch)):
            current = _layer_channel(arch, unitaries, layer, current)
            layerwise.append(current)
        outputs.append(layerwise)
    return outputs

# --------------------------------------------------------------------------- #
# Quantum filter
# --------------------------------------------------------------------------- #

class QuanvolutionFilter(tq.QuantumModule):
    """Random two‑qubit kernel applied to whole images."""

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
        self.layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))

    def forward(self, x: Tensor) -> List[qt.Qobj]:
        """Return a list of quantum states, one per batch element."""
        batch = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=batch, device=device)
        # Flatten image and take first 16 pixels for simplicity
        flat = x.view(batch, -1)[:, :16]
        states: List[qt.Qobj] = []
        for i in range(batch):
            self.encoder(qdev, flat[i])
            self.layer(qdev)
            states.append(qdev.state.copy())
        return states

# --------------------------------------------------------------------------- #
# Main quantum graph network
# --------------------------------------------------------------------------- #

class GraphQuanvolutionNet(tq.QuantumModule):
    """Hybrid quantum‑classical graph network using a quantum quanvolution filter."""

    def __init__(self, arch: Sequence[int] = (4, 8, 16), threshold: float = 0.8) -> None:
        super().__init__()
        self.arch = arch
        self.threshold = threshold

        self.qfilter = QuanvolutionFilter()

        # Randomly initialise graph‑network unitaries
        self.unitaries: List[List[qt.Qobj]] = [[_random_qubit_unitary(self.arch[0])]]
        for layer in range(1, len(self.arch)):
            ops = [_random_qubit_unitary(self.arch[layer - 1] + 1)
                   for _ in range(self.arch[layer])]
            self.unitaries.append(ops)

        # Linear head mapping qubit expectations to logits
        self.linear = nn.Linear(self.arch[-1], 10)

    def forward(self, x: Tensor) -> Tensor:
        # Quantum filter → list of states
        states = self.qfilter(x)

        # Build fidelity graph (unused in propagation but kept for analysis)
        _ = fidelity_adjacency(states, self.threshold)

        # Propagate through quantum graph
        propagated = feedforward(self.arch, self.unitaries, states)

        # Final layer states
        final_states = propagated[0][-1]

        # Compute expectation of Pauli‑Z on each qubit
        logits_list = []
        for s in final_states:
            exp_vals = [float(qt.expect(qt.sigmaz(), s.ptrace([i]))) for i in range(self.arch[-1])]
            logits_list.append(exp_vals)
        logits = torch.tensor(logits_list, dtype=torch.float32, device=x.device)

        # Linear head → log‑softmax
        logits = self.linear(logits)
        return torch.log_softmax(logits, dim=-1)

__all__ = ["GraphQuanvolutionNet"]
