"""Hybrid quantum kernel and graph utilities for graph‑based QNNs."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import itertools
import networkx as nx
import numpy as np
import qutip as qt
import scipy as sc
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
from torch import nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# 1. Variational quantum kernel ansatz
# --------------------------------------------------------------------------- #
class VariationalKernelAnsatz(tq.QuantumModule):
    """Parameterised Ry/CX ansatz that can be trained end‑to‑end."""
    def __init__(self, n_wires: int, depth: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.depth = depth
        self.params = nn.Parameter(torch.randn(depth, n_wires))
        self.op_list: List[dict] = []
        for d in range(depth):
            for w in range(n_wires):
                self.op_list.append({
                    "input_idx": [w],
                    "func": "ry",
                    "wires": [w],
                    "param_layer": d
                })
        for w in range(n_wires - 1):
            self.op_list.append({
                "input_idx": [],
                "func": "cx",
                "wires": [w, w + 1]
            })

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: Tensor, y: Tensor) -> None:
        q_device.reset_states(x.shape[0])

        for op in self.op_list:
            if op["func"] == "ry":
                idx = op["input_idx"][0]
                lam = self.params[op["param_layer"], idx]
                params = lam * x[:, idx]
                func_name_dict[op["func"]](q_device, wires=op["wires"], params=params)
            else:
                func_name_dict[op["func"]](q_device, wires=op["wires"])

        for op in reversed(self.op_list):
            if op["func"] == "ry":
                idx = op["input_idx"][0]
                lam = self.params[op["param_layer"], idx]
                params = -lam * y[:, idx]
                func_name_dict[op["func"]](q_device, wires=op["wires"], params=params)
            else:
                func_name_dict[op["func"]](q_device, wires=op["wires"])

# --------------------------------------------------------------------------- #
# 2. Quantum kernel module
# --------------------------------------------------------------------------- #
class UnifiedKernelGraphModel(tq.QuantumModule):
    """Quantum kernel and graph utilities for graph‑based QNNs.

    The module provides a variational kernel, random network generation,
    forward propagation, and fidelity‑based graph construction.
    """
    def __init__(self, n_wires: int = 4, depth: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.kernel_ansatz = VariationalKernelAnsatz(n_wires, depth)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.kernel_ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def random_network(self, arch: Sequence[int], samples: int):
        return _random_network_qnn(arch, samples)

    def feedforward(self, arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                    samples: Iterable[tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
        return _feedforward_qnn(arch, unitaries, samples)

    def state_fidelity(self, a: qt.Qobj, b: qt.Qobj) -> float:
        return _state_fidelity_qnn(a, b)

    def fidelity_adjacency(self, states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
        return _fidelity_adjacency_qnn(states, threshold,
                                       secondary=secondary,
                                       secondary_weight=secondary_weight)

# --------------------------------------------------------------------------- #
# 3. Quantum‑graph helper functions (adapted from the QML seed)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qt.Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    proj = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    proj.dims = [dims.copy(), dims.copy()]
    return proj

def _swap_registers(op: qt.Qobj, source: int, target: int) -> qt.Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

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
    amps = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    amps /= sc.linalg.norm(amps)
    state = qt.Qobj(amps)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> List[tuple[qt.Qobj, qt.Qobj]]:
    data = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        data.append((state, unitary * state))
    return data

def _random_network_qnn(qnn_arch: List[int], samples: int):
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
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def _feedforward_qnn(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                     samples: Iterable[tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def _state_fidelity_qnn(a: qt.Qobj, b: qt.Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def _fidelity_adjacency_qnn(states: Sequence[qt.Qobj], threshold: float,
                           *, secondary: float | None = None,
                           secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = _state_fidelity_qnn(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

# Expose convenient aliases
random_network = _random_network_qnn
feedforward = _feedforward_qnn
state_fidelity = _state_fidelity_qnn
fidelity_adjacency = _fidelity_adjacency_qnn

__all__ = [
    "VariationalKernelAnsatz",
    "UnifiedKernelGraphModel",
    "random_network",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
