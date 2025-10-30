"""Quantum‑centric hybrid kernel and graph neural network.

This module mirrors the classical implementation but replaces the
RBF kernel with a variational quantum kernel and the feed‑forward
network with a qutip‑based tensor‑network QNN.  The API surface
matches the original `QuantumKernelMethod.py` and `GraphQNN.py`
so that existing code can import the same symbols while gaining
quantum capabilities.

Key points
----------
* `KernalAnsatz` – a quantum data‑encoding ansatz composed of a
  user‑supplied list of gate specifications.
* `Kernel` – evaluates the overlap of two encoded states.
* `kernel_matrix` – constructs the Gram matrix on a set of
  classical vectors.
* `GraphQNN` – a tensor‑network QNN that can be trained variationally.
* `HybridKernelGraphModel` – a container that exposes both the kernel
  and the graph QNN under a unified interface.
"""

from __future__ import annotations

import itertools
import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qutip as qt
import networkx as nx
import scipy as sc
from typing import Iterable, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Quantum kernel
# --------------------------------------------------------------------------- #
class KernalAnsatz(tq.QuantumModule):
    """Programmable data‑encoding ansatz.

    Parameters
    ----------
    func_list : list[dict]
        Each dict must contain ``input_idx`` (list of feature indices),
        ``func`` (gate name understood by TorchQuantum), and ``wires``.
    """
    def __init__(self, func_list: List[dict]):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode ``x`` and undo the encoding of ``y`` on the same device."""
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Quantum kernel based on a fixed ansatz."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple Ry encoding for each qubit
        self.ansatz = KernalAnsatz(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return the absolute overlap of the first basis state
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute the Gram matrix for two collections of vectors."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------------- #
#  Quantum graph neural network utilities
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
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = sc.linalg.orth(matrix)
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> qt.Qobj:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: qt.Qobj, samples: int) -> list[tuple[qt.Qobj, qt.Qobj]]:
    dataset: list[tuple[qt.Qobj, qt.Qobj]] = []
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

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                   layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]],
                samples: Iterable[tuple[qt.Qobj, qt.Qobj]]):
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
    """Return the absolute squared overlap between pure states."""
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[qt.Qobj], threshold: float,
                       *, secondary: float | None = None,
                       secondary_weight: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class GraphQNN:
    """Tensor‑network quantum neural network that can be trained variationally."""
    def __init__(self, qnn_arch: list[int]):
        self.arch = list(qnn_arch)
        self.arch, self.unitaries, _, self.target_unitary = random_network(self.arch, samples=0)

    def forward(self, sample: qt.Qobj) -> list[qt.Qobj]:
        """Return state after each layer."""
        return feedforward(self.arch, self.unitaries, [(sample, None)])[0]

    def build_graph(self, states: Sequence[qt.Qobj], threshold: float,
                    *, secondary: float | None = None,
                    secondary_weight: float = 0.5) -> nx.Graph:
        return fidelity_adjacency(states, threshold,
                                  secondary=secondary,
                                  secondary_weight=secondary_weight)

class HybridKernelGraphModel:
    """Convenience wrapper that bundles a quantum kernel and a quantum graph QNN."""
    def __init__(self, n_wires: int = 4, qnn_arch: list[int] | None = None):
        self.kernel = Kernel(n_wires)
        self.graph_qnn = GraphQNN(qnn_arch) if qnn_arch is not None else None

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return kernel_matrix(a, b)

    def feedforward(self, sample: qt.Qobj) -> list[qt.Qobj]:
        if self.graph_qnn is None:
            raise RuntimeError("Graph QNN not initialised.")
        return self.graph_qnn.forward(sample)

    def build_graph(self, states: Sequence[qt.Qobj], threshold: float,
                    *, secondary: float | None = None,
                    secondary_weight: float = 0.5) -> nx.Graph:
        if self.graph_qnn is None:
            raise RuntimeError("Graph QNN not initialised.")
        return self.graph_qnn.build_graph(states, threshold,
                                          secondary=secondary,
                                          secondary_weight=secondary_weight)

__all__ = [
    "KernalAnsatz", "Kernel", "kernel_matrix",
    "feedforward", "fidelity_adjacency",
    "random_network", "random_training_data",
    "state_fidelity",
    "GraphQNN",
    "HybridKernelGraphModel",
]
