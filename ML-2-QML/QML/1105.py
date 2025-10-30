"""GraphQNN__gen035: Quantum utilities with a Pennylane variational circuit.

This module keeps the original QNN helpers and extends them with a
`HybridQNN` class that implements a variational circuit using Pennylane.
The class exposes a `forward_state` method that maps a classical input
vector to a quantum state and a simple `train_step` helper that
optimises the rotation angles to minimise the mean‑squared‑error loss
between the quantum output and a target classical vector.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import torch
from torch import nn

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Original seed functions (kept for compatibility)
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qml.Identity:
    return qml.Identity(num_qubits)

def _tensored_zero(num_qubits: int) -> qml.Zero:
    return qml.Zero(num_qubits)

def _swap_registers(op: qml.QubitStateVector, source: int, target: int) -> qml.QubitStateVector:
    if source == target:
        return op
    order = list(range(len(op.wires)))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> qml.QubitStateVector:
    dim = 2 ** num_qubits
    mat = torch.randn(dim, dim, dtype=torch.complex64)
    q, r = torch.linalg.qr(mat)
    d = torch.diag(r)
    ph = d / torch.abs(d)
    return q @ torch.diag(ph)

def _random_qubit_state(num_qubits: int) -> qml.QubitStateVector:
    dim = 2 ** num_qubits
    amps = torch.randn(dim, dtype=torch.complex64)
    amps = amps / torch.norm(amps)
    return qml.QubitStateVector(amps, wires=range(num_qubits))

def random_training_data(unitary: qml.QubitStateVector, samples: int) -> List[Tuple[qml.QubitStateVector, qml.QubitStateVector]]:
    dataset = []
    num_qubits = len(unitary.wires)
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary.apply(state)))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    unitaries: List[List[qml.QubitStateVector]] = [[]]
    for layer in range(1, len(qnn_arch)):
        in_f = qnn_arch[layer - 1]
        out_f = qnn_arch[layer]
        layer_ops: List[qml.QubitStateVector] = []
        for _ in range(out_f):
            op = _random_qubit_unitary(in_f + 1)
            if out_f > 1:
                op = qml.tensor(_random_qubit_unitary(in_f + 1), _tensored_id(out_f - 1))
                op = _swap_registers(op, in_f, in_f + _)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qml.QubitStateVector, keep: Sequence[int]) -> qml.QubitStateVector:
    if len(keep)!= len(state.wires):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: qml.QubitStateVector, remove: Sequence[int]) -> qml.QubitStateVector:
    keep = list(range(len(state.wires)))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qml.QubitStateVector]], layer: int, input_state: qml.QubitStateVector) -> qml.QubitStateVector:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qml.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary.apply(state), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qml.QubitStateVector]], samples: Iterable[Tuple[qml.QubitStateVector, qml.QubitStateVector]]) -> List[List[qml.QubitStateVector]]:
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qml.QubitStateVector, b: qml.QubitStateVector) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qml.QubitStateVector], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
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
# Hybrid QNN with Pennylane
# --------------------------------------------------------------------------- #
class HybridQNN:
    """A variational QNN implemented with Pennylane.

    The network architecture is specified by ``qnn_arch``.
    Each layer applies a set of rotation gates on the wires; the
    parameters are optimised to minimise a loss against a target
    classical vector.
    """

    def __init__(self, qnn_arch: Sequence[int], device_name: str = "default.qubit"):
        self.arch = list(qnn_arch)
        self.num_wires = max(self.arch)
        self.device = qml.device(device_name, wires=self.num_wires)
        # Parameters for each layer: one (num_wires, 3) tensor per layer
        self.params = [torch.nn.Parameter(torch.randn(self.num_wires, 3)) for _ in self.arch[1:]]
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _circuit(self, x: Tensor, *params) -> Tensor:
        # Pad input to num_wires
        if x.shape[0] < self.num_wires:
            x = torch.cat([x, torch.zeros(self.num_wires - x.shape[0], dtype=x.dtype)])
        qml.StatePrep(x, wires=range(self.num_wires))
        for layer_params in params:
            for wire in range(self.num_wires):
                a, b, c = layer_params[wire]
                qml.Rot(a, b, c, wires=wire)
        return qml.state()

    def forward_state(self, x: Tensor) -> Tensor:
        """Return the real part of the quantum state after the circuit."""
        return self.qnode(x, *self.params).real

    def train_step(self, optimizer: torch.optim.Optimizer, batch: Iterable[Tuple[Tensor, Tensor]], loss_fn: nn.Module = nn.MSELoss()) -> float:
        """Perform one optimisation step over a batch of (input, target) pairs."""
        self.qnode.train()
        total_loss = 0.0
        for inp, target in batch:
            optimizer.zero_grad()
            pred = self.forward_state(inp)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(batch)

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "HybridQNN",
]
