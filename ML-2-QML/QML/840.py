"""
GraphQNN__gen108: Quantum‑only graph neural network with a variational ansatz.

The module keeps the original QNN utilities but adds a training loop
that optimises the variational unitaries to reproduce target states
using a fidelity loss.  It is fully compatible with the original
seed functions and can be used as a benchmark for quantum‑classical
hybrid experiments.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import qutip as qt
import scipy as sc
import pennylane as qml
import pennylane.numpy as np

Tensor = qt.Qobj

# --------------------------------------------------------------------------- #
# Core QNN state propagation (original seed functions)
# --------------------------------------------------------------------------- #

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

def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: list[int], samples: int):
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

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[tuple[qt.Qobj, qt.Qobj]]):
    stored_states = []
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

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
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
# Hybrid quantum‑only GraphQNN class
# --------------------------------------------------------------------------- #

class GraphQNN__gen108:
    """
    Quantum‑only graph neural network that uses a variational ansatz
    for each layer.  The class provides a simple training loop
    that optimises the unitary parameters to reproduce a target state
    using a fidelity loss.  It is fully compatible with the original
    seed functions and can be used as a benchmark for quantum‑classical
    hybrid experiments.
    """

    def __init__(self, qnn_arch: List[int], device: str = "default.qubit", ansatz_depth: int = 2):
        self.qnn_arch = qnn_arch
        self.num_layers = len(qnn_arch) - 1
        self.ansatz_depth = ansatz_depth
        self.num_qubits = qnn_arch[-1]
        self.dev = qml.device(device, wires=self.num_qubits)
        # initialise variational parameters
        self.params = np.random.randn(ansatz_depth * 3 * self.num_qubits)
        self.qnode = self._create_qnode()

    def _create_qnode(self):
        @qml.qnode(self.dev, interface="numpy")
        def circuit(x, params):
            # amplitude encode the classical vector
            norm = np.linalg.norm(x)
            if norm > 0:
                x = x / norm
            qml.QubitStateVector(x, wires=range(self.num_qubits))
            idx = 0
            for _ in range(self.ansatz_depth):
                for i in range(self.num_qubits):
                    qml.RX(params[idx], wires=i); idx += 1
                    qml.RY(params[idx], wires=i); idx += 1
                    qml.RZ(params[idx], wires=i); idx += 1
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.state()
        return circuit

    def forward(self, input_state: qt.Qobj) -> qt.Qobj:
        # Convert qutip state to numpy vector
        vec = input_state.full().flatten()
        out_vec = self.qnode(vec, self.params)
        return qt.Qobj(out_vec.reshape(2 ** self.num_qubits, 1))

    def fidelity_loss(self, target: qt.Qobj, output: qt.Qobj) -> float:
        return abs((target.dag() * output)[0, 0]) ** 2

    def train_quantum(
        self,
        training_data: List[Tuple[qt.Qobj, qt.Qobj]],
        epochs: int = 20,
        lr: float = 0.01,
    ) -> None:
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for epoch in range(epochs):
            def cost(p):
                loss = 0.0
                for input_state, target_state in training_data:
                    out_vec = self.qnode(input_state.full().flatten(), p)
                    out_state = qt.Qobj(out_vec.reshape(2 ** self.num_qubits, 1))
                    loss += 1 - self.fidelity_loss(target_state, out_state)
                return loss / len(training_data)
            self.params = opt.step(cost, self.params)
            loss_val = cost(self.params)
            print(f"Epoch {epoch+1}/{epochs} - loss: {loss_val:.4f}")

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN__gen108",
]
