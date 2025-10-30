"""Quantum graph neural network using qutip and Pennylane.

This module implements the quantum side of GraphQNN, providing:
- Random quantum network generation with amplitude‑encoded states.
- Quantum feedforward through unitary layers.
- Fidelity‑based adjacency graph.
- QCNN variational circuit built with Pennylane.
"""

import itertools
from typing import Iterable, Sequence, Tuple, List, Union, Optional
import networkx as nx
import qutip as qt
import pennylane as qml
import numpy as np

QState = qt.Qobj


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
    unitary = np.linalg.svd(matrix, full_matrices=False)[0]
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


def random_training_data(unitary: qt.Qobj, samples: int) -> List[Tuple[qt.Qobj, qt.Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset


def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[List[qt.Qobj]], List[Tuple[qt.Qobj, qt.Qobj]], qt.Qobj]:
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


def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], layer: int, input_state: qt.Qobj) -> qt.Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))

    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary

    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))


def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[qt.Qobj]], samples: Iterable[Tuple[qt.Qobj, qt.Qobj]]) -> List[List[qt.Qobj]]:
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


def vector_to_state(vec: np.ndarray) -> qt.Qobj:
    vec = vec / np.linalg.norm(vec)
    dim = int(np.log2(len(vec)))
    return qt.Qobj(vec.reshape(-1, 1), dims=[[2] * dim, [1] * dim])


def state_to_vector(state: qt.Qobj) -> np.ndarray:
    return state.full().reshape(-1)


def QCNN(num_qubits: int = 8, depth: int = 2) -> qml.QNode:
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface="autograd")
    def circuit(inputs: np.ndarray):
        # Feature map: encode input amplitudes
        for i, val in enumerate(inputs):
            qml.RY(val, wires=i)
        # Variational layers
        for _ in range(depth):
            for i in range(num_qubits):
                qml.RX(np.random.uniform(0, 2*np.pi), wires=i)
                qml.CNOT(wires=[i, (i+1)%num_qubits])
        return qml.expval(qml.PauliZ(0))
    return circuit


class GraphQNN:
    """Hybrid quantum graph neural network."""
    def __init__(self, arch: Sequence[int], use_qnn: bool = True):
        self.arch = list(arch)
        self.use_qnn = use_qnn
        self.unitaries: List[List[qt.Qobj]] = []
        self.training_data: List[Tuple[qt.Qobj, qt.Qobj]] = []
        self.target: qt.Qobj = None

    def build_random(self, samples: int):
        self.arch, self.unitaries, self.training_data, self.target = random_network(self.arch, samples)

    def encode_graph(self, graph: nx.Graph) -> qt.Qobj:
        """Amplitude encode adjacency matrix into a quantum state."""
        adj = nx.to_numpy_array(graph)
        vec = adj.flatten()
        return vector_to_state(vec)

    def forward(self, state: qt.Qobj) -> List[qt.Qobj]:
        return feedforward(self.arch, self.unitaries, [(state, None)])

    def hybrid_forward(self, graph: nx.Graph) -> List[qt.Qobj]:
        state = self.encode_graph(graph)
        return self.forward(state)

    def train(self, epochs: int = 20, lr: float = 0.01):
        """Simple COBYLA-based training for quantum parameters."""
        if not self.unitaries:
            raise RuntimeError("Unitary layers not initialized.")
        # Flatten all unitary matrices into parameters
        params = np.concatenate([u.full().ravel().real for layer in self.unitaries for u in layer])
        def objective(p):
            idx = 0
            for layer in self.unitaries:
                for u in layer:
                    dim = u.shape[0]
                    mat = p[idx:idx+dim*dim].reshape(dim, dim)
                    u.__setitem__(slice(None), qt.Qobj(mat))
                    idx += dim*dim
            loss = 0.0
            for inp, tgt in self.training_data:
                out = self.forward(inp)[-1]
                loss += (out.full() - tgt.full()).norm() ** 2
            return loss / len(self.training_data)
        from scipy.optimize import minimize
        res = minimize(objective, params, method="COBYLA", options={"maxiter": epochs})
        idx = 0
        for layer in self.unitaries:
            for u in layer:
                dim = u.shape[0]
                mat = res.x[idx:idx+dim*dim].reshape(dim, dim)
                u.__setitem__(slice(None), qt.Qobj(mat))
                idx += dim*dim

    @staticmethod
    def QCNN(num_qubits: int = 8, depth: int = 2) -> qml.QNode:
        return QCNN(num_qubits, depth)


__all__ = [
    "feedforward",
    "fidelity_adjacency",
    "random_network",
    "random_training_data",
    "state_fidelity",
    "GraphQNN",
]
