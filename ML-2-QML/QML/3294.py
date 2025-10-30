"""Hybrid quantum graph neural network that mirrors the classical API.

The module builds upon the original GraphQNN quantum utilities by adding
a quantum kernel ansatz and a graph construction based on state
fidelities.  The class `HybridGraphQNN` is a `torchquantum.QuantumModule`
and can be used interchangeably with the classical `HybridGraphQNN`
defined in the ML module.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, List, Tuple

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import networkx as nx
import qutip as qt

Tensor = torch.Tensor

# --------------------------------------------------------------------------- #
# Random unitary and state utilities
# --------------------------------------------------------------------------- #

def _tensored_id(num_qubits: int) -> qt.Qobj:
    """Identity operator as a Qobj."""
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> qt.Qobj:
    """Zero projector for ancilla qubits."""
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
    """Generate a random unitary on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    """Sample a random pure state on `num_qubits` qubits."""
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

# --------------------------------------------------------------------------- #
# Training data generation
# --------------------------------------------------------------------------- #

def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    """Generate (state, U|state>) pairs for a fixed unitary."""
    dataset: List[Tuple[qt.Qobj, qt.Qobj]] = []
    num_qubits = int(np.log2(unitary.shape[0]))
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: Sequence[int], samples: int):
    """Create a random quantum network and associated training data."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: List[List[qt.Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[qt.Qobj] = []
        for _ in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return list(qnn_arch), unitaries, training_data, target_unitary

# --------------------------------------------------------------------------- #
# Partial trace utilities
# --------------------------------------------------------------------------- #

def _partial_trace_keep(state: qt.Qobj, keep: Sequence[int]) -> qt.Qobj:
    """Partial trace over all qubits except those in `keep`."""
    total = int(np.log2(state.shape[0]))
    if len(keep) == total:
        return state
    return state.ptrace(list(keep))

def _partial_trace_remove(state: qt.Qobj, remove: Sequence[int]) -> qt.Qobj:
    keep = list(range(int(np.log2(state.shape[0]))))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

# --------------------------------------------------------------------------- #
# Layer channel
# --------------------------------------------------------------------------- #

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    layer: int,
    input_state: qt.Qobj,
) -> qt.Qobj:
    """Apply a layer of unitaries and trace out the input qubits."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

# --------------------------------------------------------------------------- #
# Feedforward
# --------------------------------------------------------------------------- #

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[qt.Qobj]],
    samples: Iterable[Tuple[qt.Qobj, qt.Qobj]],
) -> List[List[qt.Qobj]]:
    """Propagate each sample through the quantum network."""
    stored_states: List[List[qt.Qobj]] = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

# --------------------------------------------------------------------------- #
# Fidelity utilities
# --------------------------------------------------------------------------- #

def state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
    """Overlap squared between two pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(
    states: Sequence[qt.Qobj],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    """Weighted graph from state fidelities."""
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
# Quantum kernel utilities
# --------------------------------------------------------------------------- #

class KernalAnsatz(tq.QuantumModule):
    """Encode two classical vectors into a quantum circuit and compute overlap."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: Tensor, y: Tensor) -> None:
        """Apply the encoding of x followed by the inverse encoding of y."""
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Return the absolute overlap between the encoded states."""
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[Tensor], b: Sequence[Tensor]) -> np.ndarray:
    """Compute the Gram matrix between two data sets using the quantum kernel."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
# Hybrid quantum graph neural network module
# --------------------------------------------------------------------------- #

class HybridGraphQNN(tq.QuantumModule):
    """
    Quantum analogue of the classical HybridGraphQNN.  The class
    exposes the same public API (forward, build_graph, compute_kernel_matrix)
    so that it can be swapped in place of the classical version.
    """
    def __init__(self, arch: Sequence[int]) -> None:
        super().__init__()
        self.arch = list(arch)
        self.n_wires = max(self.arch)
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Randomly initialise a unitary per layer
        self.unitaries: List[List[qt.Qobj]] = [[]]
        for layer in range(1, len(self.arch)):
            num_inputs = self.arch[layer - 1]
            num_outputs = self.arch[layer]
            layer_ops: List[qt.Qobj] = []
            for _ in range(num_outputs):
                op = _random_qubit_unitary(num_inputs + 1)
                layer_ops.append(op)
            self.unitaries.append(layer_ops)

    def forward(self, x: qt.Qobj) -> List[qt.Qobj]:
        """Return the list of intermediate quantum states for a single input."""
        states = [x]
        current = x
        for layer in range(1, len(self.arch)):
            current = _layer_channel(self.arch, self.unitaries, layer, current)
            states.append(current)
        return states

    def build_graph(self, states: Sequence[qt.Qobj], threshold: float) -> nx.Graph:
        """Construct a graph from state fidelities."""
        return fidelity_adjacency(states, threshold)

    def compute_kernel_matrix(self, a: Sequence[qt.Qobj], b: Sequence[qt.Qobj]) -> np.ndarray:
        """Return the Gram matrix using the quantum kernel ansatz."""
        return kernel_matrix(a, b)

__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "kernel_matrix",
    "HybridGraphQNN",
]
