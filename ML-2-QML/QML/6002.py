"""Quantum graph neural network utilities.

This module mirrors the seed's quantum implementation but
exposes a dedicated `QuantumGraphQNN` class that can be trained
with PyTorch autograd.  PennyLane's GPU‑accelerated
`default.qubit.jit` device is used for efficient simulation.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import List, Tuple

import networkx as nx
import pennylane as qml
import torch

# --------------------------------------------------------------------------- #
#  Core QNN state propagation
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> qml.QubitOperator:
    return qml.Identity(num_qubits)

def _tensored_zero(num_qubits: int) -> qml.QubitOperator:
    return qml.Identity(num_qubits)

def _swap_registers(op: qml.QubitOperator, source: int, target: int) -> qml.QubitOperator:
    if source == target:
        return op
    swap_gate = qml.SWAP(source, target)
    return swap_gate @ op

def _random_qubit_unitary(num_qubits: int) -> qml.QubitOperator:
    dim = 2 ** num_qubits
    matrix = torch.randn(dim, dim, dtype=torch.complex128)
    q, _ = torch.linalg.qr(matrix)
    return qml.QubitOperator.from_matrix(q, wires=range(num_qubits))

def _random_qubit_state(num_qubits: int) -> qml.QubitOperator:
    dim = 2 ** num_qubits
    vec = torch.randn(dim, dtype=torch.complex128)
    vec = vec / torch.norm(vec)
    return qml.QubitOperator.from_matrix(vec.reshape(-1, 1), wires=range(num_qubits))

def random_training_data(unitary: qml.QubitOperator, samples: int) -> List[Tuple[qml.QubitOperator, qml.QubitOperator]]:
    dataset: List[Tuple[qml.QubitOperator, qml.QubitOperator]] = []
    num_qubits = unitary.wires
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary @ state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qml.QubitOperator]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qml.QubitOperator] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: qml.QubitOperator, keep: Sequence[int]) -> qml.QubitOperator:
    return state.partial_trace(keep)

def _partial_trace_remove(state: qml.QubitOperator, remove: Sequence[int]) -> qml.QubitOperator:
    keep = list(range(state.wires))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qml.QubitOperator]],
    layer: int,
    input_state: qml.QubitOperator,
) -> qml.QubitOperator:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qml.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0]
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary
    return _partial_trace_remove(layer_unitary @ state @ layer_unitary.adjoint(), range(num_inputs))

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qml.QubitOperator]],
    samples: Iterable[Tuple[qml.QubitOperator, qml.QubitOperator]],
) -> List[List[qml.QubitOperator]]:
    stored_states: List[List[qml.QubitOperator]] = []
    for sample, _ in samples:
        layerwise: List[qml.QubitOperator] = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: qml.QubitOperator, b: qml.QubitOperator) -> float:
    overlap = (a.adjoint() @ b).coeffs[0].real
    return abs(overlap) ** 2

def fidelity_adjacency(
    states: Sequence[qml.QubitOperator],
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

# --------------------------------------------------------------------------- #
#  Quantum Graph Neural Network
# --------------------------------------------------------------------------- #
class QuantumGraphQNN:
    """A simple hybrid quantum graph neural network built with PennyLane.

    Parameters
    ----------
    arch : Sequence[int]
        Node counts per layer including input and output.
    """

    def __init__(self, arch: Sequence[int]):
        self.arch = list(arch)
        self.layers = []
        for in_f, out_f in zip(self.arch[:-1], self.arch[1:]):
            self.layers.append(QuantumLayer(in_f, out_f))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def train_step(
        self,
        data_loader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.modules.loss._Loss,
    ) -> None:
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

class QuantumLayer:
    """Variational quantum layer implemented with PennyLane.

    The layer encodes the input vector as Y rotations, applies a
    parameterised unitary on all qubits, and measures the Z
    expectation of the output qubits.
    """

    def __init__(self, in_qubits: int, out_qubits: int):
        self.in_qubits = in_qubits
        self.out_qubits = out_qubits
        self.total_qubits = in_qubits + out_qubits
        self.dev = qml.device("default.qubit.jit", wires=self.total_qubits)
        # Parameters for the variational circuit
        self.params = torch.randn(out_qubits, self.total_qubits, dtype=torch.complex128, requires_grad=True)
        self.qnode = qml.QNode(self._circuit, self.dev, diff_method="adjoint")

    def _circuit(self, params: torch.Tensor, x: torch.Tensor) -> List[float]:
        for i in range(self.in_qubits):
            qml.RY(x[i], wires=i)
        for out in range(self.out_qubits):
            for w in range(self.total_qubits):
                qml.RX(params[out, w], wires=w)
        return [qml.expval(qml.PauliZ(i)) for i in range(self.in_qubits, self.total_qubits)]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        batch_out = []
        for i in range(x.shape[0]):
            out = self.qnode(self.params, x[i])
            batch_out.append(out)
        return torch.tensor(batch_out, dtype=torch.float64)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "QuantumGraphQNN",
    "QuantumLayer",
]
