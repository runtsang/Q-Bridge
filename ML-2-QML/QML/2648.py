"""HybridSelfAttentionGraphQNN: quantum self‑attention + quantum graph utilities."""
from __future__ import annotations

import itertools
import numpy as np
import qiskit
import qutip as qt
import networkx as nx
from typing import Iterable, Sequence, List, Tuple

Qobj = qt.Qobj

# --------------------------------------------------------------------------- #
# Helper functions for quantum graph utilities
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> Qobj:
    identity = qt.qeye(2 ** num_qubits)
    dims = [2] * num_qubits
    identity.dims = [dims.copy(), dims.copy()]
    return identity

def _tensored_zero(num_qubits: int) -> Qobj:
    projector = qt.fock(2 ** num_qubits).proj()
    dims = [2] * num_qubits
    projector.dims = [dims.copy(), dims.copy()]
    return projector

def _swap_registers(op: Qobj, source: int, target: int) -> Qobj:
    if source == target:
        return op
    order = list(range(len(op.dims[0])))
    order[source], order[target] = order[target], order[source]
    return op.permute(order)

def _random_qubit_unitary(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    matrix = np.random.normal(size=(dim, dim)) + 1j * np.random.normal(size=(dim, dim))
    unitary = np.linalg.qr(matrix)[0]
    qobj = qt.Qobj(unitary)
    dims = [2] * num_qubits
    qobj.dims = [dims.copy(), dims.copy()]
    return qobj

def _random_qubit_state(num_qubits: int) -> Qobj:
    dim = 2 ** num_qubits
    amplitudes = np.random.normal(size=(dim, 1)) + 1j * np.random.normal(size=(dim, 1))
    amplitudes /= np.linalg.norm(amplitudes)
    state = qt.Qobj(amplitudes)
    state.dims = [[2] * num_qubits, [1] * num_qubits]
    return state

def random_training_data(unitary: Qobj, samples: int) -> List[Tuple[Qobj, Qobj]]:
    dataset = []
    num_qubits = len(unitary.dims[0])
    for _ in range(samples):
        state = _random_qubit_state(num_qubits)
        dataset.append((state, unitary * state))
    return dataset

def random_network(qnn_arch: List[int], samples: int):
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)
    unitaries: List[List[Qobj]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: List[Qobj] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = qt.tensor(_random_qubit_unitary(num_inputs + 1), _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)
    return qnn_arch, unitaries, training_data, target_unitary

def _partial_trace_keep(state: Qobj, keep: Sequence[int]) -> Qobj:
    if len(keep)!= len(state.dims[0]):
        return state.ptrace(list(keep))
    return state

def _partial_trace_remove(state: Qobj, remove: Sequence[int]) -> Qobj:
    keep = list(range(len(state.dims[0])))
    for index in sorted(remove, reverse=True):
        keep.pop(index)
    return _partial_trace_keep(state, keep)

def _layer_channel(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]],
                   layer: int, input_state: Qobj) -> Qobj:
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = qt.tensor(input_state, _tensored_zero(num_outputs))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate * layer_unitary
    return _partial_trace_remove(layer_unitary * state * layer_unitary.dag(), range(num_inputs))

def feedforward(qnn_arch: Sequence[int], unitaries: Sequence[Sequence[Qobj]],
                samples: Iterable[Tuple[Qobj, Qobj]]) -> List[List[Qobj]]:
    stored_states = []
    for sample, _ in samples:
        layerwise = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, unitaries, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: Qobj, b: Qobj) -> float:
    return abs((a.dag() * b)[0, 0]) ** 2

def fidelity_adjacency(states: Sequence[Qobj], threshold: float,
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

# --------------------------------------------------------------------------- #
# Quantum self‑attention circuit
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Variational circuit that outputs a probability vector interpretable as attention weights."""
    def __init__(self, n_qubits: int, backend=None):
        self.n_qubits = n_qubits
        self.backend = backend or qiskit.Aer.get_backend('qasm_simulator')
        self.qr = qiskit.QuantumRegister(n_qubits, 'q')
        self.cr = qiskit.ClassicalRegister(n_qubits, 'c')

    def _build_circuit(self, rot: np.ndarray, ent: np.ndarray) -> qiskit.QuantumCircuit:
        circ = qiskit.QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rot[3 * i], i)
            circ.ry(rot[3 * i + 1], i)
            circ.rz(rot[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(ent[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def run(self, rot: np.ndarray, ent: np.ndarray, shots: int = 1024):
        circ = self._build_circuit(rot, ent)
        job = qiskit.execute(circ, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(circ)
        total = sum(counts.values())
        probs = np.array([counts.get(format(i, f'0{self.n_qubits}b'), 0) / total for i in range(2 ** self.n_qubits)])
        return probs

# --------------------------------------------------------------------------- #
# Hybrid orchestrator (quantum side)
# --------------------------------------------------------------------------- #
class HybridSelfAttentionGraphQNN:
    """Quantum‑centric orchestrator that combines self‑attention circuit with quantum graph propagation."""
    def __init__(self, n_qubits: int, qnn_arch: Sequence[int]):
        self.attention = QuantumSelfAttention(n_qubits)
        self.qnn_arch = list(qnn_arch)
        _, self.unitaries, self.training_data, self.target_unitary = random_network(qnn_arch, samples=10)

    def run_attention(self, rot: np.ndarray, ent: np.ndarray, shots: int = 1024):
        return self.attention.run(rot, ent, shots)

    def run_graph(self, samples: Iterable[Tuple[Qobj, Qobj]]) -> List[List[Qobj]]:
        return feedforward(self.qnn_arch, self.unitaries, samples)

    def build_fidelity_graph(self, states: Sequence[Qobj], threshold: float) -> nx.Graph:
        return fidelity_adjacency(states, threshold)

__all__ = ["HybridSelfAttentionGraphQNN", "QuantumSelfAttention", "random_network",
           "random_training_data", "feedforward", "state_fidelity", "fidelity_adjacency"]
