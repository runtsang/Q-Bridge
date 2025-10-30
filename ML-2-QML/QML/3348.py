"""Hybrid quantum graph‑based classifier using a variational circuit."""

from __future__ import annotations

import itertools
from typing import Iterable, List, Tuple

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

def build_classifier_circuit(num_qubits: int, depth: int, adjacency: nx.Graph | None = None) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """Construct a variational circuit whose entanglement pattern follows an adjacency graph."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    # Data encoding
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    w_idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[w_idx], qubit)
            w_idx += 1
        if adjacency is None:
            # Default chain connectivity
            for qubit in range(num_qubits - 1):
                circuit.cz(qubit, qubit + 1)
        else:
            for u, v in adjacency.edges():
                circuit.cz(u, v)

    # Observables: single‑qubit Z operators
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

def random_training_data(unitary: QuantumCircuit, samples: int) -> List[Tuple[QuantumCircuit, QuantumCircuit]]:
    """Generate random input states and the target state after applying a unitary."""
    dataset: List[Tuple[QuantumCircuit, QuantumCircuit]] = []
    for _ in range(samples):
        qc = QuantumCircuit(unitary.num_qubits)
        for q in range(unitary.num_qubits):
            qc.rx(np.random.rand() * 2 * np.pi, q)
            qc.ry(np.random.rand() * 2 * np.pi, q)
            qc.rz(np.random.rand() * 2 * np.pi, q)
        target = qc.compose(unitary)
        dataset.append((qc, target))
    return dataset

def random_network(qnn_arch: List[int], samples: int) -> Tuple[List[int], List[QuantumCircuit], List[Tuple[QuantumCircuit, QuantumCircuit]], QuantumCircuit]:
    """Create a random layered quantum circuit and training data."""
    num_qubits = qnn_arch[-1]
    # Target unitary is a random single‑qubit rotation per output qubit
    target_unitary = QuantumCircuit(num_qubits)
    for q in range(num_qubits):
        target_unitary.rx(np.random.rand() * 2 * np.pi, q)
        target_unitary.ry(np.random.rand() * 2 * np.pi, q)
        target_unitary.rz(np.random.rand() * 2 * np.pi, q)
    training_data = random_training_data(target_unitary, samples)

    # Intermediate layer circuits
    circuits: List[QuantumCircuit] = []
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_circ = QuantumCircuit(num_outputs)
        for q in range(num_outputs):
            layer_circ.rx(np.random.rand() * 2 * np.pi, q)
            layer_circ.ry(np.random.rand() * 2 * np.pi, q)
            layer_circ.rz(np.random.rand() * 2 * np.pi, q)
        circuits.append(layer_circ)

    return qnn_arch, circuits, training_data, target_unitary

def _layer_channel(qnn_arch: List[int], circuits: List[QuantumCircuit], layer: int, input_state: QuantumCircuit) -> QuantumCircuit:
    """Apply a single layer circuit to an input state."""
    return circuits[layer - 1]  # simplified: returns the layer circuit

def feedforward(qnn_arch: List[int], circuits: List[QuantumCircuit], samples: Iterable[Tuple[QuantumCircuit, QuantumCircuit]]) -> List[List[QuantumCircuit]]:
    """Forward pass through the layered quantum circuit."""
    stored_states: List[List[QuantumCircuit]] = []
    for sample, _ in samples:
        layerwise: List[QuantumCircuit] = [sample]
        current_state = sample
        for layer in range(1, len(qnn_arch)):
            current_state = _layer_channel(qnn_arch, circuits, layer, current_state)
            layerwise.append(current_state)
        stored_states.append(layerwise)
    return stored_states

def state_fidelity(a: QuantumCircuit, b: QuantumCircuit) -> float:
    """Compute fidelity between two pure states using a statevector simulator."""
    sv_a = Statevector(a)
    sv_b = Statevector(b)
    return abs(np.vdot(sv_a.data, sv_b.data)) ** 2

def fidelity_adjacency(states: List[QuantumCircuit], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
    """Construct an adjacency graph from state fidelities."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
        fid = state_fidelity(state_i, state_j)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph

class HybridGraphClassifier:
    """Quantum graph‑based classifier with a variational circuit."""
    def __init__(self, num_qubits: int, depth: int, adjacency: nx.Graph | None = None):
        self.num_qubits = num_qubits
        self.depth = depth
        self.adjacency = adjacency
        self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(num_qubits, depth, adjacency)

    def evaluate(self, params: List[float]) -> List[float]:
        """Return expectation values of observables for a given parameter set."""
        param_list = [p for pv in self.encoding + self.weights for p in pv]
        bound = self.circuit.bind_parameters(dict(zip(param_list, params)))
        backend = Aer.get_backend("statevector_simulator")
        job = execute(bound, backend)
        result = job.result()
        statevector = result.get_statevector(bound)
        exp_vals: List[float] = []
        for obs in self.observables:
            exp_vals.append(np.real(np.vdot(statevector, obs.to_matrix() @ statevector)))
        return exp_vals
